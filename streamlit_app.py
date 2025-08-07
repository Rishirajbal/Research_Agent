"""
RishiGPT Research Assistant - Streamlit Frontend
"""

import streamlit as st
import uuid
import os
import tempfile
import base64
from typing import Dict, Any, Optional
import markdown
from dotenv import load_dotenv
from backend import RishiGPTConfig, RishiGPTResearcher

load_dotenv()


def markdown_to_html(markdown_text):
    html = markdown.markdown(markdown_text)
    return html


def create_simple_pdf_content(markdown_text):
    html_content = markdown.markdown(markdown_text)
    full_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1f77b4; }}
            h2 {{ color: #333; margin-top: 30px; }}
            h3 {{ color: #666; }}
            p {{ line-height: 1.6; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>RishiGPT Research Report</h1>
            <p>Generated on {uuid.uuid4().hex[:8]}</p>
        </div>
        {html_content}
    </body>
    </html>
    """
    return full_html


def create_pdf_from_markdown(markdown_text):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import io
        
        # Create a buffer to store the PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor='#1f77b4'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor='#333333'
        )
        
        # Convert markdown to simple text (basic conversion)
        lines = markdown_text.split('\n')
        story = []
        
        # Add title
        story.append(Paragraph("RishiGPT Research Report", title_style))
        story.append(Spacer(1, 20))
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Handle headers
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                if level == 1:
                    story.append(Paragraph(text, title_style))
                else:
                    story.append(Paragraph(text, heading_style))
                story.append(Spacer(1, 12))
            elif line.startswith('##'):
                # Handle subheaders
                text = line.lstrip('#').strip()
                story.append(Paragraph(text, heading_style))
                story.append(Spacer(1, 12))
            elif line:
                # Handle regular text
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        return None


def main():
    st.set_page_config(
        page_title="RishiGPT Research",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analyst-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">RishiGPT Research</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Research Assistant</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("API Keys")
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Enter your Groq API key"
        )
        
        tavily_api_key = st.text_input(
            "Tavily API Key", 
            type="password",
            help="Enter your Tavily API key"
        )
        
        st.subheader("Model Settings")
        analyst_model = st.selectbox(
            "Analyst Model",
            ["meta-llama/llama-4-scout-17b-16e-instruct", "llama3-70b-8192", "mixtral-8x7b-32768"],
            help="Model for generating analysts"
        )
        
        research_model = st.selectbox(
            "Research Model",
            ["moonshotai/kimi-k2-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "llama3-70b-8192", "mixtral-8x7b-32768"],
            help="Model for conducting research"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Control randomness in responses"
        )
        
        st.subheader("Generated Analysts")
        if "analysts" in st.session_state:
            for i, analyst in enumerate(st.session_state.analysts, 1):
                with st.expander(f"Analyst {i}: {analyst.name}"):
                    st.write(f"**Role:** {analyst.role}")
                    st.write(f"**Affiliation:** {analyst.affiliation}")
                    st.write(f"**Description:** {analyst.description}")
        else:
            st.info("No analysts generated yet. Start research to see analysts here.")
    
    config = RishiGPTConfig()
    config.analyst_model_name = analyst_model
    config.research_model_name = research_model
    config.temperature = temperature
    config.max_analysts = 3
    config.max_interview_turns = 2
    
    if groq_api_key:
        config.set_groq_api_key(groq_api_key)
    if tavily_api_key:
        config.set_tavily_api_key(tavily_api_key)
    
    if not config.validate_config():
        st.error("Please provide both Groq API Key and Tavily API Key in the sidebar")
        return
    
    try:
        researcher = RishiGPTResearcher(config)
    except Exception as e:
        st.error(f"Failed to initialize researcher: {str(e)}")
        return
    
    tab1, tab2, tab3 = st.tabs(["Start Research", "Final Report", "Process Visualization"])
    
    with tab1:
        st.header("Start New Research")
        
        topic = st.text_area(
            "Research Topic",
            placeholder="Enter your research topic here...",
            height=100,
            help="Describe what you want to research"
        )
        
        if st.button("Start Research", type="primary", disabled=not topic.strip()):
            if topic.strip():
                with st.spinner("Starting research process..."):
                    try:
                        thread_id = str(uuid.uuid4())
                        
                        # Start research and complete it in one go
                        result = researcher.start_research(
                            topic=topic.strip(),
                            max_analysts=3,
                            thread_id=thread_id
                        )
                        
                        if result["status"] == "analysts_generated":
                            st.success("Analysts generated! Starting research...")
                            
                            # Store analysts in session state for sidebar display
                            st.session_state.analysts = result["analysts"]
                            
                            # Continue research to completion
                            research_result = researcher.confirm_analysts(result["thread"])
                            
                            if research_result["status"] == "research_complete":
                                st.success("Research completed successfully!")
                                st.session_state.final_report = research_result["final_report"]
                                st.session_state.topic = topic.strip()
                                st.rerun()
                            else:
                                st.error(f"Research failed: {research_result.get('message', 'Unknown error')}")
                        else:
                            st.error(f"{result.get('message', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"Error during research: {str(e)}")
    
    with tab2:
        st.header("Final Research Report")
        
        if "final_report" in st.session_state:
            st.success("Research completed!")
            
            st.markdown(st.session_state.final_report)
            
            report_text = st.session_state.final_report
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Report as Markdown",
                    data=report_text,
                    file_name=f"rishigpt_research_report_{uuid.uuid4().hex[:8]}.md",
                    mime="text/markdown"
                )
            
            with col2:
                try:
                    # Try PDF first
                    pdf_bytes = create_pdf_from_markdown(report_text)
                    if pdf_bytes:
                        st.download_button(
                            label="Download Report as PDF",
                            data=pdf_bytes,
                            file_name=f"rishigpt_research_report_{uuid.uuid4().hex[:8]}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        # Fallback to HTML
                        html_content = create_simple_pdf_content(report_text)
                        st.download_button(
                            label="Download Report as HTML",
                            data=html_content,
                            file_name=f"rishigpt_research_report_{uuid.uuid4().hex[:8]}.html",
                            mime="text/html"
                        )
                        st.info("HTML file can be opened in a browser and printed as PDF")
                except Exception as e:
                    st.error(f"Error generating download: {str(e)}")
            
            st.subheader("Research Process Visualization")
            try:
                graph_html = researcher.get_graph_html()
                st.components.v1.html(graph_html, height=600)
            except Exception as e:
                st.error(f"Error displaying graph: {str(e)}")
        else:
            st.info("No final report available. Complete a research project to see the report here.")
    
    with tab3:
        st.header("Research Process Visualization")
        st.write("This visualization shows the flow of the research process and how different components interact.")
        
        try:
            graph_html = researcher.get_graph_html()
            st.components.v1.html(graph_html, height=700)
            
            st.subheader("Process Explanation")
            st.markdown("""
            The research process follows these main steps:
            
            1. **Create Analysts**: The system generates AI analysts with different perspectives based on the research topic.
            2. **Human Feedback**: You can provide feedback to refine the analysts or confirm them to proceed.
            3. **Conduct Interviews**: Each analyst conducts an interview with an expert, searching for information online.
            4. **Write Reports**: The system generates section reports, introduction, and conclusion based on the interviews.
            5. **Finalize Report**: All components are combined into a final research report.
            
            The dashed lines represent the interview subprocess, which includes:
            - Asking questions
            - Searching for information (web and Wikipedia)
            - Answering questions
            - Saving the interview
            - Writing a section based on the interview
            """)
        except Exception as e:
            st.error(f"Error displaying graph: {str(e)}")


if __name__ == "__main__":
    main()
