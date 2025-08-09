import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import json

# Configure page
st.set_page_config(
    page_title="FinPlan GenAI Platform MVP Demo",
    page_icon="🤖",
    layout="wide"
)

# Mock data for demo
MOCK_PROMPTS = {
    "Budget Analysis": "Analyze this user's budget and provide insights on spending patterns. Focus on areas for improvement.",
    "Investment Advice": "Based on this portfolio data, provide personalized investment recommendations considering risk tolerance.",
    "Loan Recommendation": "Evaluate this user's financial profile and recommend appropriate loan products.",
    "Credit Card Analysis": "Compare these credit card offers and recommend the best option for this user's spending habits.",
    "Expense Categorization": "Categorize these transactions and identify any unusual spending patterns."
}

MOCK_TEAMS = ["Budget Team", "Investment Team", "Lending Team", "Credit Team", "Goals Team", "Integrations Team"]

# Generate mock usage data
@st.cache_data
def generate_mock_data():
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    usage_data = []
    cost_data = []
    
    for date in dates:
        for team in MOCK_TEAMS:
            requests = random.randint(50, 500)
            cost = requests * random.uniform(0.002, 0.008)
            
            usage_data.append({
                "date": date,
                "team": team,
                "requests": requests,
                "avg_latency": random.uniform(80, 150),
                "error_rate": random.uniform(0.1, 2.0)
            })
            
            cost_data.append({
                "date": date,
                "team": team,
                "cost": cost,
                "model": random.choice(["GPT-4", "Claude-3.5", "GPT-3.5"])
            })
    
    return pd.DataFrame(usage_data), pd.DataFrame(cost_data)

def simulate_pii_scan(text):
    """Mock PII detection"""
    pii_patterns = ["123-45-6789", "ssn", "social security", "account number", "routing number"]
    detected = [pattern for pattern in pii_patterns if pattern.lower() in text.lower()]
    return detected

def simulate_bias_check(text):
    """Mock bias detection"""
    bias_indicators = ["always", "never", "guaranteed", "risk-free", "certain"]
    detected = [word for word in bias_indicators if word.lower() in text.lower()]
    return detected

# Main app
def main():
    st.title("🤖 FinPlan GenAI Platform MVP Demo")
    st.markdown("*Internal GenAI platform consolidating monitoring, compliance, and infrastructure*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.selectbox("Select Feature", [
        "🚀 Smart API Gateway",
        "📚 Prompt Library", 
        "🔒 Auto-Compliance",
        "📊 Usage Dashboard",
        "💻 SDK Integration"
    ])
    
    if tab == "🚀 Smart API Gateway":
        show_api_gateway()
    elif tab == "📚 Prompt Library":
        show_prompt_library()
    elif tab == "🔒 Auto-Compliance":
        show_compliance_module()
    elif tab == "📊 Usage Dashboard":
        show_dashboard()
    elif tab == "💻 SDK Integration":
        show_sdk_demo()

def show_api_gateway():
    st.header("🚀 Smart API Gateway")
    st.markdown("*Unified API for all LLM providers with automatic routing, rate limiting, and cost tracking*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Make a Request")
        
        # Team selection
        team = st.selectbox("Select your team:", MOCK_TEAMS)
        
        # Model selection (optional)
        model = st.selectbox("Preferred model (optional):", ["Auto", "GPT-4", "Claude-3.5", "GPT-3.5"])
        
        # Input
        user_input = st.text_area("Enter your prompt:", 
                                 "Analyze this user's monthly spending: Groceries $600, Rent $2000, Entertainment $300, Utilities $150")
        
        if st.button("Send Request", type="primary"):
            with st.spinner("Processing request..."):
                # Simulate API call delay
                time.sleep(2)
                
                # Mock response
                response = f"""Based on the spending analysis for this user:

**Budget Breakdown:**
- Housing (Rent): $2,000 (65% of total)
- Food (Groceries): $600 (19% of total) 
- Entertainment: $300 (10% of total)
- Utilities: $150 (5% of total)
- **Total Monthly Spending: $3,050**

**Key Insights:**
- Housing costs are high at 65% of spending - typically recommended to stay under 50%
- Grocery spending is reasonable at 19%
- Entertainment budget could be optimized

**Recommendations:**
- Consider finding lower-cost housing options
- Set up automatic savings to build emergency fund
- Track discretionary spending more closely"""
                
                st.success("✅ Request completed successfully!")
                st.markdown("**Response:**")
                st.markdown(response)
    
    with col2:
        st.subheader("Request Details")
        if 'response' in locals():
            st.metric("Latency", "127ms")
            st.metric("Model Used", "GPT-4" if model == "Auto" else model)
            st.metric("Tokens", "456")
            st.metric("Cost", "$0.0034")
            
            # Request routing info
            st.markdown("**Routing Info:**")
            st.code(f"""
Team: {team}
Provider: OpenAI
Region: us-east-1
Cache: MISS
Retry: 0
            """)

def show_prompt_library():
    st.header("📚 Prompt Library")
    st.markdown("*Centralized repository of tested prompts for financial use cases*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Browse Prompts")
        
        # Search
        search = st.text_input("🔍 Search prompts:", placeholder="e.g., budget, investment")
        
        # Category filter
        category = st.selectbox("Category:", ["All", "Budget", "Investment", "Lending", "Credit", "Analysis"])
        
        # Prompt list
        filtered_prompts = MOCK_PROMPTS
        if search:
            filtered_prompts = {k: v for k, v in MOCK_PROMPTS.items() if search.lower() in k.lower()}
        
        selected_prompt = st.radio("Select a prompt:", list(filtered_prompts.keys()))
    
    with col2:
        st.subheader("Prompt Details")
        
        if selected_prompt:
            st.markdown(f"**{selected_prompt}**")
            st.code(filtered_prompts[selected_prompt])
            
            # Prompt metadata
            st.markdown("**Metadata:**")
            metadata_df = pd.DataFrame([
                {"Field": "Version", "Value": "v2.1"},
                {"Field": "Success Rate", "Value": "94.2%"},
                {"Field": "Avg Latency", "Value": "156ms"},
                {"Field": "Last Updated", "Value": "2025-07-15"},
                {"Field": "Usage Count", "Value": "1,247"},
                {"Field": "Teams Using", "Value": "4"}
            ])
            st.dataframe(metadata_df, hide_index=True)
            
            # Test prompt
            st.markdown("**Test this prompt:**")
            test_input = st.text_area("Sample data:", "Sample financial data here...")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Test Prompt"):
                    st.info("✅ Prompt test successful - 98ms response time")
            with col_b:
                if st.button("Fork Prompt"):
                    st.success("✅ Prompt forked to your team's workspace")

def show_compliance_module():
    st.header("🔒 Auto-Compliance Scanner")
    st.markdown("*Automatic PII detection and bias monitoring for financial AI*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PII Detection")
        
        pii_input = st.text_area("Enter text to scan for PII:", 
                                value="Customer John Smith (SSN: 123-45-6789) has account number 9876543210 and wants a loan.")
        
        if st.button("Scan for PII"):
            pii_detected = simulate_pii_scan(pii_input)
            
            if pii_detected:
                st.error(f"⚠️ PII Detected: {', '.join(pii_detected)}")
                st.markdown("**Redacted version:**")
                redacted = pii_input.replace("123-45-6789", "***-**-****").replace("9876543210", "**********")
                st.code(redacted)
            else:
                st.success("✅ No PII detected - text is safe to process")
    
    with col2:
        st.subheader("Bias Detection")
        
        bias_input = st.text_area("Enter financial advice to check for bias:", 
                                 value="This investment is guaranteed to make money and is completely risk-free for everyone.")
        
        if st.button("Check for Bias"):
            bias_detected = simulate_bias_check(bias_input)
            
            if bias_detected:
                st.warning(f"⚠️ Potential bias detected: {', '.join(bias_detected)}")
                st.markdown("**Recommendations:**")
                st.markdown("- Avoid absolute terms like 'guaranteed' or 'risk-free'")
                st.markdown("- Include appropriate disclaimers about investment risks")
                st.markdown("- Consider individual circumstances and risk tolerance")
            else:
                st.success("✅ No obvious bias detected")
    
    # Compliance dashboard
    st.subheader("Compliance Summary")
    
    compliance_metrics = pd.DataFrame([
        {"Metric": "Requests Scanned Today", "Value": "2,847", "Status": "✅"},
        {"Metric": "PII Detections", "Value": "23", "Status": "⚠️"},
        {"Metric": "Bias Alerts", "Value": "7", "Status": "⚠️"},
        {"Metric": "Blocked Requests", "Value": "3", "Status": "🚫"},
        {"Metric": "Compliance Score", "Value": "98.9%", "Status": "✅"}
    ])
    st.dataframe(compliance_metrics, hide_index=True)

def show_dashboard():
    st.header("📊 Usage Dashboard")
    st.markdown("*Real-time monitoring of GenAI usage across all teams*")
    
    # Generate mock data
    usage_df, cost_df = generate_mock_data()
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_requests = usage_df.groupby('date')['requests'].sum().iloc[-1]
    total_cost = cost_df.groupby('date')['cost'].sum().iloc[-1]
    avg_latency = usage_df.groupby('date')['avg_latency'].mean().iloc[-1]
    error_rate = usage_df.groupby('date')['error_rate'].mean().iloc[-1]
    
    col1.metric("Daily Requests", f"{total_requests:,}", delta="12%")
    col2.metric("Daily Cost", f"${total_cost:.2f}", delta="-5%")
    col3.metric("Avg Latency", f"{avg_latency:.0f}ms", delta="3ms")
    col4.metric("Error Rate", f"{error_rate:.1f}%", delta="-0.2%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Usage by Team (Last 7 Days)")
        recent_usage = usage_df[usage_df['date'] >= datetime.now() - timedelta(days=7)]
        team_usage = recent_usage.groupby('team')['requests'].sum().reset_index()
        
        fig = px.pie(team_usage, values='requests', names='team', 
                     title="Request Distribution by Team")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cost Trends (Last 30 Days)")
        daily_costs = cost_df.groupby('date')['cost'].sum().reset_index()
        
        fig = px.line(daily_costs, x='date', y='cost', 
                      title="Daily GenAI Costs")
        fig.update_layout(xaxis_title="Date", yaxis_title="Cost ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Team Performance Details")
    
    team_summary = usage_df.groupby('team').agg({
        'requests': 'sum',
        'avg_latency': 'mean',
        'error_rate': 'mean'
    }).reset_index()
    
    team_costs = cost_df.groupby('team')['cost'].sum().reset_index()
    team_summary = team_summary.merge(team_costs, on='team')
    
    team_summary.columns = ['Team', 'Total Requests', 'Avg Latency (ms)', 'Error Rate (%)', 'Total Cost ($)']
    team_summary['Avg Latency (ms)'] = team_summary['Avg Latency (ms)'].round(0)
    team_summary['Error Rate (%)'] = team_summary['Error Rate (%)'].round(2)
    team_summary['Total Cost ($)'] = team_summary['Total Cost ($)'].round(2)
    
    st.dataframe(team_summary, hide_index=True)

def show_sdk_demo():
    st.header("💻 SDK Integration Demo")
    st.markdown("*Simple SDK for easy GenAI integration*")
    
    # Installation
    st.subheader("1. Installation")
    st.code("pip install finplan-genai-platform", language="bash")
    
    # Basic usage
    st.subheader("2. Basic Usage")
    st.code("""
from finplan_genai import GenAIClient

# Initialize client (automatically uses your SSO credentials)
client = GenAIClient()

# Simple chat completion
response = client.chat(
    prompt="Analyze this budget for overspending",
    data={"rent": 2000, "groceries": 600, "entertainment": 300}
)

print(response.content)
# Output: "Your housing costs are 65% of spending, which exceeds the recommended 50%..."
    """, language="python")
    
    # Advanced usage
    st.subheader("3. Advanced Features")
    
    tab1, tab2, tab3 = st.tabs(["Prompt Templates", "Cost Tracking", "Compliance"])
    
    with tab1:
        st.code("""
# Use prompt templates from library
response = client.chat_with_template(
    template="budget_analysis_v2",
    user_data=budget_data,
    team="budget_team"
)

# A/B testing prompts
response = client.chat_with_variant(
    template="investment_advice",
    variant="conservative_v2",  # or "aggressive_v1"
    user_data=portfolio_data
)
        """, language="python")
    
    with tab2:
        st.code("""
# Track costs by feature
response = client.chat(
    prompt="Investment recommendation",
    data=user_data,
    cost_tracking={
        "feature": "portfolio_advisor",
        "user_id": "user_12345"
    }
)

# Get cost summary
costs = client.get_costs(
    team="investment_team",
    date_range="last_30_days"
)
        """, language="python")
    
    with tab3:
        st.code("""
# Automatic compliance checking
response = client.chat(
    prompt="Generate financial advice",
    data=user_data,
    compliance={
        "pii_detection": True,
        "bias_monitoring": True,
        "audit_logging": True
    }
)

# Check compliance status
if response.compliance_issues:
    print(f"Issues detected: {response.compliance_issues}")
        """, language="python")
    
    # SDK Demo Interactive
    st.subheader("4. Try the SDK")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input Parameters:**")
        demo_team = st.selectbox("Team:", MOCK_TEAMS, key="sdk_team")
        demo_prompt = st.text_area("Prompt:", "Analyze spending patterns", key="sdk_prompt")
        demo_data = st.text_area("User Data (JSON):", '{"spending": {"rent": 2000, "food": 600}}', key="sdk_data")
    
    with col2:
        st.markdown("**Generated Code:**")
        generated_code = f"""
from finplan_genai import GenAIClient

client = GenAIClient(team="{demo_team}")

response = client.chat(
    prompt="{demo_prompt}",
    data={demo_data}
)

print(response.content)
print(f"Cost: ${{response.cost}}")
print(f"Latency: {{response.latency}}ms")
        """
        st.code(generated_code, language="python")
        
        if st.button("Execute Demo Code"):
            with st.spinner("Executing..."):
                time.sleep(1.5)
            st.success("✅ Code executed successfully!")
            st.markdown("**Output:**")
            st.code("""
Housing costs at 77% of total spending exceed recommended limits. Consider reducing rent or increasing income.

Cost: $0.0023
Latency: 142ms
            """)

if __name__ == "__main__":
    main()
