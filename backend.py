"""
RishiGPT Research Assistant - Backend Logic
"""

import os
import functools
import operator
import json
from typing import Dict, Any, Optional, List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network

load_dotenv()


class RishiGPTConfig:
    def __init__(self):
        # Initialize with None, will be set by user input
        self.groq_api_key = None
        self.tavily_api_key = None
        self.langsmith_api_key = None
        self.analyst_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.research_model_name = "moonshotai/kimi-k2-instruct"
        self.temperature = 0.1
        self.max_analysts = 3
        self.max_interview_turns = 2

    def set_groq_api_key(self, api_key: str):
        self.groq_api_key = api_key
        os.environ["GROQ_API_KEY"] = api_key

    def set_tavily_api_key(self, api_key: str):
        self.tavily_api_key = api_key
        os.environ["TAVILY_API_KEY"] = api_key

    def set_langsmith_api_key(self, api_key: str):
        self.langsmith_api_key = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        if api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = "rishigpt"

    def validate_config(self) -> bool:
        return self.groq_api_key is not None and self.tavily_api_key is not None

    def get_missing_keys(self) -> List[str]:
        missing = []
        if not self.groq_api_key:
            missing.append("Groq API Key")
        if not self.tavily_api_key:
            missing.append("Tavily API Key")
        return missing


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


def create_llm(config: RishiGPTConfig, model_name: str = None) -> ChatGroq:
    if model_name is None:
        model_name = config.research_model_name
    return ChatGroq(
        model=model_name,
        temperature=config.temperature
    )


def create_structured_llm(config: RishiGPTConfig, output_schema):
    llm = create_llm(config)
    return llm.with_structured_output(output_schema)


ANALYST_INSTRUCTIONS = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""


QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


SEARCH_INSTRUCTIONS = """You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""


ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""


SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""


INTRO_CONCLUSION_INSTRUCTIONS = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""


def create_analysts(state: GenerateAnalystsState, config: RishiGPTConfig):
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    llm = create_llm(config, config.analyst_model_name)
    
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback, 
        max_analysts=max_analysts
    )
    
    try:
        structured_llm = create_structured_llm(config, Perspectives)
        analysts = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts.")
        ])
        return {"analysts": analysts.analysts}
    except Exception as e:
        # Fallback to regular LLM if structured output fails
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts in JSON format with a list of analysts, each having name, role, affiliation, and description fields.")
        ])
        
        # Simple parsing fallback
        import json
        import re
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                analysts_data = json.loads(json_match.group())
                analysts = []
                for analyst_data in analysts_data:
                    analyst = Analyst(
                        name=analyst_data.get('name', 'Unknown'),
                        role=analyst_data.get('role', 'Unknown'),
                        affiliation=analyst_data.get('affiliation', 'Unknown'),
                        description=analyst_data.get('description', 'Unknown')
                    )
                    analysts.append(analyst)
                return {"analysts": analysts}
        except:
            pass
        
        # Create default analysts if parsing fails
        default_analysts = [
            Analyst(
                name="Research Analyst 1",
                role="General Analyst",
                affiliation="Research Institute",
                description="Focuses on general research aspects and methodology."
            ),
            Analyst(
                name="Research Analyst 2", 
                role="Technical Analyst",
                affiliation="Technical Institute",
                description="Specializes in technical and implementation details."
            ),
            Analyst(
                name="Research Analyst 3",
                role="Business Analyst", 
                affiliation="Business School",
                description="Focuses on business implications and market analysis."
            )
        ]
        return {"analysts": default_analysts[:max_analysts]}


def human_feedback(state: GenerateAnalystsState):
    pass


def should_continue(state: GenerateAnalystsState):
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return "END"


def generate_question(state: InterviewState, config: RishiGPTConfig):
    analyst = state["analyst"]
    messages = state["messages"]
    
    system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
    llm = create_llm(config)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {"messages": [question]}


def search_web(state: InterviewState, config: RishiGPTConfig):
    try:
        structured_llm = create_structured_llm(config, SearchQuery)
        search_query = structured_llm.invoke([SystemMessage(content=SEARCH_INSTRUCTIONS)] + state['messages'])
        query = search_query.search_query
    except:
        # Fallback to regular LLM
        llm = create_llm(config)
        response = llm.invoke([SystemMessage(content=SEARCH_INSTRUCTIONS)] + state['messages'])
        query = response.content.strip()
    
    try:
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(query)
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return {"context": [formatted_search_docs]}
    except Exception as e:
        # Return empty context if search fails
        return {"context": ["<Document>Search failed. No additional context available.</Document>"]}


def search_wikipedia(state: InterviewState, config: RishiGPTConfig):
    try:
        structured_llm = create_structured_llm(config, SearchQuery)
        search_query = structured_llm.invoke([SystemMessage(content=SEARCH_INSTRUCTIONS)] + state['messages'])
        query = search_query.search_query
    except:
        # Fallback to regular LLM
        llm = create_llm(config)
        response = llm.invoke([SystemMessage(content=SEARCH_INSTRUCTIONS)] + state['messages'])
        query = response.content.strip()
    
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return {"context": [formatted_search_docs]}
    except Exception as e:
        # Return empty context if search fails
        return {"context": ["<Document>Wikipedia search failed. No additional context available.</Document>"]}


def generate_answer(state: InterviewState, config: RishiGPTConfig):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]
    
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
    llm = create_llm(config)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    answer.name = "expert"
    
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)
    
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    
    if num_responses >= max_num_turns:
        return 'save_interview'
    
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"


def write_section(state: InterviewState, config: RishiGPTConfig):
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
    
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
    llm = create_llm(config)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {context}")
    ])
    
    return {"sections": [section.content]}


def initiate_all_interviews(state: ResearchGraphState, config: RishiGPTConfig):
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
            }) for analyst in state["analysts"]
        ]


def write_report(state: ResearchGraphState, config: RishiGPTConfig):
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    system_message = REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections)
    llm = create_llm(config)
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos.")
    ])
    return {"content": report.content}


def write_introduction(state: ResearchGraphState, config: RishiGPTConfig):
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    llm = create_llm(config)
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report introduction")
    ])
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState, config: RishiGPTConfig):
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    llm = create_llm(config)
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report conclusion")
    ])
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None
    
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


class RishiGPTResearcher:
    def __init__(self, config: RishiGPTConfig):
        self.config = config
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.graph_html = self._visualize_graph()
    
    def _build_graph(self):
        analyst_builder = StateGraph(GenerateAnalystsState)
        analyst_builder.add_node("create_analysts", 
                               lambda state: create_analysts(state, self.config))
        analyst_builder.add_node("human_feedback", human_feedback)
        analyst_builder.add_edge(START, "create_analysts")
        analyst_builder.add_edge("create_analysts", "human_feedback")
        analyst_builder.add_conditional_edges("human_feedback", should_continue, 
                                           ["create_analysts", END])
        
        analyst_graph = analyst_builder.compile(
            interrupt_before=['human_feedback'], 
            checkpointer=self.memory
        )
        
        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", 
                                  lambda state: generate_question(state, self.config))
        interview_builder.add_node("search_web", 
                                  lambda state: search_web(state, self.config))
        interview_builder.add_node("search_wikipedia", 
                                  lambda state: search_wikipedia(state, self.config))
        interview_builder.add_node("answer_question", 
                                  lambda state: generate_answer(state, self.config))
        interview_builder.add_node("save_interview", save_interview)
        interview_builder.add_node("write_section", 
                                 lambda state: write_section(state, self.config))
        
        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", route_messages,
                                              ['ask_question', 'save_interview'])
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)
        
        interview_graph = interview_builder.compile(checkpointer=self.memory)
        
        builder = StateGraph(ResearchGraphState)
        builder.add_node("create_analysts", 
                        lambda state: create_analysts(state, self.config))
        builder.add_node("human_feedback", human_feedback)
        builder.add_node("conduct_interview", interview_graph)
        builder.add_node("write_report", 
                        lambda state: write_report(state, self.config))
        builder.add_node("write_introduction", 
                        lambda state: write_introduction(state, self.config))
        builder.add_node("write_conclusion", 
                        lambda state: write_conclusion(state, self.config))
        builder.add_node("finalize_report", finalize_report)
        
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", 
                                   lambda state: initiate_all_interviews(state, self.config),
                                   ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], 
                        "finalize_report")
        builder.add_edge("finalize_report", END)
        
        return builder.compile(interrupt_before=['human_feedback'], checkpointer=self.memory)
    
    def _visualize_graph(self):
        try:
            # Create a simplified representation of the graph for visualization
            G = nx.DiGraph()
            
            # Main research flow
            G.add_node("START", title="Start", color="#6FCF97")
            G.add_node("create_analysts", title="Create Analysts", color="#56CCF2")
            G.add_node("human_feedback", title="Human Feedback", color="#F2C94C")
            G.add_node("conduct_interview", title="Conduct Interview", color="#BB6BD9")
            G.add_node("write_report", title="Write Report", color="#F2994A")
            G.add_node("write_introduction", title="Write Introduction", color="#F2994A")
            G.add_node("write_conclusion", title="Write Conclusion", color="#F2994A")
            G.add_node("finalize_report", title="Finalize Report", color="#EB5757")
            G.add_node("END", title="End", color="#6FCF97")
            
            # Interview flow
            G.add_node("ask_question", title="Ask Question", color="#BB6BD9")
            G.add_node("search_web", title="Search Web", color="#56CCF2")
            G.add_node("search_wikipedia", title="Search Wikipedia", color="#56CCF2")
            G.add_node("answer_question", title="Answer Question", color="#BB6BD9")
            G.add_node("save_interview", title="Save Interview", color="#F2C94C")
            G.add_node("write_section", title="Write Section", color="#F2994A")
            
            # Main research edges
            G.add_edge("START", "create_analysts")
            G.add_edge("create_analysts", "human_feedback")
            G.add_edge("human_feedback", "create_analysts", label="feedback")
            G.add_edge("human_feedback", "conduct_interview", label="confirm")
            G.add_edge("conduct_interview", "write_report")
            G.add_edge("conduct_interview", "write_introduction")
            G.add_edge("conduct_interview", "write_conclusion")
            G.add_edge("write_report", "finalize_report")
            G.add_edge("write_introduction", "finalize_report")
            G.add_edge("write_conclusion", "finalize_report")
            G.add_edge("finalize_report", "END")
            
            # Interview edges
            G.add_edge("conduct_interview", "ask_question", style="dashed")
            G.add_edge("ask_question", "search_web", style="dashed")
            G.add_edge("ask_question", "search_wikipedia", style="dashed")
            G.add_edge("search_web", "answer_question", style="dashed")
            G.add_edge("search_wikipedia", "answer_question", style="dashed")
            G.add_edge("answer_question", "ask_question", label="continue", style="dashed")
            G.add_edge("answer_question", "save_interview", label="done", style="dashed")
            G.add_edge("save_interview", "write_section", style="dashed")
            
            # Create a PyVis network
            net = Network(height="600px", width="100%", directed=True, notebook=False)
            
            # Add nodes
            for node in G.nodes():
                net.add_node(node, 
                            label=node, 
                            title=G.nodes[node].get('title', node),
                            color=G.nodes[node].get('color', '#97C2FC'))
            
            # Add edges
            for edge in G.edges():
                net.add_edge(edge[0], edge[1], 
                            title=G.edges[edge].get('label', ''),
                            label=G.edges[edge].get('label', ''),
                            style=G.edges[edge].get('style', 'solid'))
            
            # Generate HTML
            net.set_options("""
            var options = {
                "nodes": {
                    "font": {
                        "size": 14,
                        "face": "Tahoma"
                    },
                    "shape": "box"
                },
                "edges": {
                    "arrows": {
                        "to": {
                            "enabled": true
                        }
                    },
                    "smooth": {
                        "type": "curvedCW",
                        "roundness": 0.2
                    }
                },
                "physics": {
                    "hierarchicalRepulsion": {
                        "centralGravity": 0.5,
                        "springLength": 150,
                        "springConstant": 0.01,
                        "nodeDistance": 120,
                        "damping": 0.09
                    },
                    "solver": "hierarchicalRepulsion"
                },
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "LR",
                        "sortMethod": "directed",
                        "levelSeparation": 150
                    }
                }
            }
            """)
            
            # Save to a temporary file
            graph_html = "graph.html"
            net.save_graph(graph_html)
            
            # Read the HTML content
            with open(graph_html, 'r') as f:
                html_content = f.read()
            
            return html_content
        except Exception as e:
            print(f"Error visualizing graph: {str(e)}")
            return "<p>Graph visualization failed</p>"
    
    def start_research(self, topic: str, max_analysts: int = 3, thread_id: str = "1") -> Dict[str, Any]:
        thread = {"configurable": {"thread_id": thread_id}}
        
        for event in self.graph.stream(
            {"topic": topic, "max_analysts": max_analysts}, 
            thread, 
            stream_mode="values"
        ):
            analysts = event.get('analysts', '')
            if analysts:
                return {
                    "status": "analysts_generated",
                    "analysts": analysts,
                    "thread": thread
                }
        
        return {"status": "error", "message": "Failed to generate analysts"}
    
    def provide_feedback(self, feedback: str, thread: Dict[str, Any]) -> Dict[str, Any]:
        self.graph.update_state(
            thread, 
            {"human_analyst_feedback": feedback}, 
            as_node="human_feedback"
        )
        
        for event in self.graph.stream(None, thread, stream_mode="values"):
            analysts = event.get('analysts', '')
            if analysts:
                return {
                    "status": "analysts_updated",
                    "analysts": analysts,
                    "thread": thread
                }
        
        return {"status": "error", "message": "Failed to update analysts"}
    
    def confirm_analysts(self, thread: Dict[str, Any]) -> Dict[str, Any]:
        self.graph.update_state(
            thread, 
            {"human_analyst_feedback": None}, 
            as_node="human_feedback"
        )
        
        for event in self.graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            if node_name == "finalize_report":
                final_state = self.graph.get_state(thread)
                return {
                    "status": "research_complete",
                    "final_report": final_state.values.get('final_report'),
                    "thread": thread
                }
        
        return {"status": "error", "message": "Failed to complete research"}
    
    def get_current_analysts(self, thread: Dict[str, Any]) -> Optional[list]:
        try:
            state = self.graph.get_state(thread)
            return state.values.get('analysts')
        except:
            return None
    
    def get_final_report(self, thread: Dict[str, Any]) -> Optional[str]:
        try:
            state = self.graph.get_state(thread)
            return state.values.get('final_report')
        except:
            return None
            
    def get_graph_html(self) -> str:
        return self.graph_html
