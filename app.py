import streamlit as st
import pandas as pd
import os
import json
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import uuid
import pandas as pd
# Import our updated NL2SsQL processor and data summary utility
try:
    from groqnopenai import SmartNL2SQLProcessor
except ImportError:
    st.error("Please ensure 'groqnopenai.py' is in the same directory as this Streamlit app")
    st.stop()
    
# Configure Streamlit page
st.set_page_config(
    page_title="Smart Business Data Analytics - Chat Interface",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4CAF50;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #1f77b4;
    }
    
    .error-message {
        background-color: #fff5f5;
        border-left-color: #ff6b6b;
    }
    
    .warning-message {
        background-color: #fffbf0;
        border-left-color: #ffa500;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    
    .main-content {
        padding-bottom: 120px;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        width: 100%;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #4CAF50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .welcome-message {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .welcome-message h3 {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .welcome-message ul {
        margin-left: 1rem;
    }
    
    .welcome-message li {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.chat_sessions[st.session_state.current_session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'title': 'New Chat'
        }
    
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

@st.cache_resource
def get_processor():
    """Initialize and cache the NL2SQL processor"""
    try:
        return SmartNL2SQLProcessor()
    except Exception as e:
        st.error(f"Failed to initialize processor: {str(e)}")
        return None

def create_new_chat():
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_session_id
    st.session_state.chat_sessions[new_session_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'title': 'New Chat'
    }
    st.session_state.query_input = ""

def get_current_session():
    """Get current chat session"""
    if st.session_state.current_session_id not in st.session_state.chat_sessions:
        create_new_chat()
    return st.session_state.chat_sessions[st.session_state.current_session_id]

def add_message_to_session(message_type, content, **kwargs):
    """Add a message to the current chat session"""
    session = get_current_session()
    message = {
        'type': message_type,
        'content': content,
        'timestamp': datetime.now(),
        **kwargs
    }
    session['messages'].append(message)
    
    # Update session title based on first message
    if len(session['messages']) == 1 and message_type == 'user':
        session['title'] = content[:30] + "..." if len(content) > 30 else content

        
def display_chat_message(message):
    """Display a chat message with appropriate styling"""

    # Convert timestamp string to datetime if needed
    timestamp_val = message.get('timestamp')
    if isinstance(timestamp_val, str):
        try:
            timestamp = datetime.fromisoformat(timestamp_val)
        except Exception:
            # Fallback: display the raw string timestamp if conversion fails
            timestamp = timestamp_val
    else:
        timestamp = timestamp_val

    if isinstance(timestamp, datetime):
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_str = str(timestamp)

    if message['type'] == 'assistant':
        if message.get('error'):
            # Display error message in red
            st.markdown(f"""<div style="color: red; border: 1px solid red; padding: 10px; border-radius: 5px;">
            <b>Error:</b> {message['content']}<br><small>{time_str}</small></div>""", unsafe_allow_html=True)
        elif message.get('warning'):
            # Display warning message in orange
            st.markdown(f"""<div style="color: orange; border: 1px solid orange; padding: 10px; border-radius: 5px;">
            <b>Warning:</b> {message['content']}<br><small>{time_str}</small></div>""", unsafe_allow_html=True)
        else:
            # Success/normal assistant message
            st.markdown(f"""<div style="color: green; border: 1px solid green; padding: 10px; border-radius: 5px;">
            {message['content']}<br><small>{time_str}</small></div>""", unsafe_allow_html=True)

        # Display metrics if available
        if message.get('execution_results') is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records Found", len(message['execution_results']))
            with col2:
                st.metric("Processing Time", f"{message.get('processing_time_seconds', 0):.2f}s")
            with col3:
                st.metric("Visualization", "Generated" if message.get('graph_file') else "None")

        # Display graph if available
        graph_file = message.get('graph_file')
        if graph_file and os.path.exists(graph_file):
            st.markdown(f"**{message.get('graph_type', 'Visualization')}:**")
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)
            except Exception as e:
                st.error(f"Error displaying chart: {e}")

        # Display data table if result set is small
        if message.get('execution_results') and len(message['execution_results']) <= 20:
            with st.expander("View Data Table"):
                try:
                    df = pd.DataFrame(message['execution_results'])
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying data table: {e}")

        # Show SQL query in expander if available
        if message.get('sql_query'):
            with st.expander("🔍 View Generated SQL"):
                st.code(message['sql_query'], language='sql')

    elif message['type'] == 'user':
        # Display user message
        st.markdown(f"""<div style="background-color: #cce5ff; padding: 10px; border-radius: 5px;">
        <b>User:</b> {message['content']}<br><small>{time_str}</small></div>""", unsafe_allow_html=True)

    else:
        # For any other message types, just display content plainly
        st.markdown(f"{message['content']}<br><small>{time_str}</small>", unsafe_allow_html=True)


def display_chat_history():
    """Display the chat history for current session"""
    session = get_current_session()
    
    if not session['messages']:
        st.markdown("""
        <div class="welcome-message">
            <h3>Welcome to Smart Business Analytics!</h3>
            <p>I'm here to help you analyze your business data. Ask me questions about:</p>
            <ul>
                <li>📈 Sales trends and performance</li>
                <li>👥 Employee attendance and productivity</li>
                <li>📦 Inventory and product analysis</li>
                <li>💰 Financial metrics and comparisons</li>
                <li>🏪 Outlet and regional performance</li>
            </ul>
            <p><strong>Try asking:</strong> "Show me monthly sales trends" or "Which products are performing best?"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in session['messages']:
            display_chat_message(message)

# Initialize session state
initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Smart Business Data Analytics</h1>
    <p>AI-Powered Chat Interface for Business Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💬 Chat Management")
    
    # System Status
    processor = get_processor()
    if processor:
        st.success("✅ System Ready")
    else:
        st.error("❌ System Error")
        st.stop()
    
    # New Chat Button
    if st.button("🆕 Start New Chat", type="primary"):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat Sessions
    st.header("📚 Chat History")
    if st.session_state.chat_sessions:
        # Sort sessions by creation time (newest first)
        sorted_sessions = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        for session_id, session_data in sorted_sessions[:10]:  # Show last 10 sessions
            is_current = session_id == st.session_state.current_session_id
            button_text = f"{'🔵' if is_current else '⚪'} {session_data['title']}"
            
            if st.button(button_text, key=f"session_{session_id}", disabled=is_current):
                st.session_state.current_session_id = session_id
                st.rerun()
    else:
        st.info("No chat history yet")
    
    st.markdown("---")
    
    # Sample queries
    st.header("💡 Sample Queries")
    sample_queries = [
        "Show total sales by category",
        "Monthly sales trend for 2024", 
        "Attendance rate by department",
        "Revenue analysis by outlet",
        "Which products are top performers?",
        "Compare sales across regions"
    ]
    
    for query in sample_queries:
        if st.button(f"📝 {query}", key=f"sample_{query}"):
            st.session_state.query_input = query
    
    st.markdown("---")
    
    # Session Info
    current_session = get_current_session()
    st.header("ℹ️ Current Session")
    st.write(f"**Messages:** {len(current_session['messages'])}")
    st.write(f"**Started:** {current_session['created_at'].strftime('%H:%M')}")

# Main content area
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Chat history display
st.markdown("### 💬 Conversation")
display_chat_history()

st.markdown('</div>', unsafe_allow_html=True)

# Fixed input at bottom
st.markdown("---")
st.markdown("### 🔍 Ask Your Question")

# Query input form
with st.form(key="query_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Type your business question here:",
            value=st.session_state.query_input,
            placeholder="Example: Show me sales trends for the last quarter by product category",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        submitted = st.form_submit_button("🚀 Send", type="primary")

# Process query
if submitted and user_query.strip():
    # Reset the input
    st.session_state.query_input = ""
    
    # Add user message to chat
    add_message_to_session('user', user_query)
    
    # Show processing indicator
    with st.spinner('🤖 Analyzing your query...'):
        try:
            # Get previous context for follow-up
            session = get_current_session()
            previous_messages = session['messages'][-6:]  # Last 6 messages for context
            
            context = ""
            if len(previous_messages) > 1:
                context = "Previous conversation context:\n"
                for msg in previous_messages[:-1]:  # Exclude current message
                    if msg['type'] == 'user':
                        context += f"User asked: {msg['content']}\n"
                    elif msg['type'] == 'assistant' and msg.get('description'):
                        context += f"Assistant found: {msg['description'][:100]}...\n"
            
            # Process the query
            result = processor.process_query_with_smart_visualization(user_query, context)
            
            # Add assistant response to chat
            add_message_to_session('assistant', result.get('description', ''), **result)
            
        except Exception as e:
            error_result = {
                'error': f"An unexpected error occurred: {str(e)}",
                'description': "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
            }
            add_message_to_session('assistant', error_result.get('description', ''), **error_result)
    
    # Rerun to show new messages
    st.rerun()

elif submitted and not user_query.strip():
    st.warning("⚠️ Please enter a question before sending.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    🤖 Powered by AI • 📊 Smart Analytics • 🔒 Secure Processing<br>
    💡 Tip: You can ask follow-up questions in the same chat or start a new chat for different topics
</div>
""", unsafe_allow_html=True)