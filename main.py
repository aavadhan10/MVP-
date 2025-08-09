import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import json
import uuid

# Configure page
st.set_page_config(
    page_title="GenAI Dev Core MVP Demo",
    page_icon="🛠️",
    layout="wide"
)

# Mock data for the evaluation framework
EVALUATION_METRICS = {
    "accuracy_score": {"min": 0, "max": 100, "unit": "%", "threshold": 85},
    "hallucination_risk": {"min": 0, "max": 100, "unit": "%", "threshold": 10},
    "safety_score": {"min": 0, "max": 100, "unit": "%", "threshold": 90},
    "compliance_score": {"min": 0, "max": 100, "unit": "%", "threshold": 95},
    "response_latency": {"min": 0, "max": 5000, "unit": "ms", "threshold": 2000}
}

FINANCIAL_COMPLIANCE_RULES = [
    {"rule": "PII_Detection", "description": "Detects SSN, account numbers, personal identifiers"},
    {"rule": "Risk_Disclosure", "description": "Ensures investment advice includes risk warnings"},
    {"rule": "Bias_Check", "description": "Flags discriminatory language in financial advice"},
    {"rule": "Accuracy_Claims", "description": "Prevents unsupported accuracy/guarantee claims"},
    {"rule": "GDPR_Compliance", "description": "Checks for proper data handling disclosures"}
]

SAMPLE_OUTPUTS = [
    {
        "id": "out_001",
        "team": "Budget Team",
        "model": "GPT-4",
        "input": "Analyze my spending: rent $2000, food $600, entertainment $400",
        "output": "Your housing costs at 50% of income are reasonable. Consider reducing entertainment spending and building an emergency fund.",
        "timestamp": datetime.now() - timedelta(hours=2),
        "evaluation_status": "completed"
    },
    {
        "id": "out_002", 
        "team": "Investment Team",
        "model": "Claude-3.5",
        "input": "Should I invest in cryptocurrency?",
        "output": "Cryptocurrency can be part of a diversified portfolio, but should not exceed 5-10% of total investments due to volatility. Consider your risk tolerance and investment timeline.",
        "timestamp": datetime.now() - timedelta(hours=1),
        "evaluation_status": "flagged"
    },
    {
        "id": "out_003",
        "team": "Lending Team", 
        "model": "GPT-3.5",
        "input": "What loan amount can I qualify for with 650 credit score?",
        "output": "With a 650 credit score, you may qualify for loans but at higher interest rates. Exact amounts depend on income, debt-to-income ratio, and lender policies.",
        "timestamp": datetime.now() - timedelta(minutes=30),
        "evaluation_status": "passed"
    }
]

@st.cache_data
def generate_monitoring_data():
    """Generate mock monitoring data for the dashboard"""
    dates = [datetime.now() - timedelta(hours=x) for x in range(24, 0, -1)]
    
    monitoring_data = []
    for date in dates:
        for team in ["Budget Team", "Investment Team", "Lending Team", "Credit Team"]:
            requests = random.randint(20, 100)
            failure_rate = random.uniform(0.5, 3.0)
            avg_latency = random.uniform(800, 2500)
            
            monitoring_data.append({
                "timestamp": date,
                "team": team,
                "requests": requests,
                "failure_rate": failure_rate,
                "avg_latency": avg_latency,
                "accuracy_score": random.uniform(85, 95),
                "safety_score": random.uniform(88, 98),
                "compliance_violations": random.randint(0, 5)
            })
    
    return pd.DataFrame(monitoring_data)

def evaluate_output(model_output, evaluation_type="full"):
    """Simulate the evaluation framework"""
    
    # Simulate evaluation processing time
    time.sleep(1.5)
    
    # Mock evaluation results
    results = {
        "accuracy_score": random.uniform(75, 95),
        "hallucination_risk": random.uniform(2, 15),
        "safety_score": random.uniform(85, 98),
        "compliance_score": random.uniform(80, 98),
        "response_latency": random.uniform(800, 2200)
    }
    
    # Check for specific compliance issues in the text
    compliance_issues = []
    if "guaranteed" in model_output.lower() or "risk-free" in model_output.lower():
        compliance_issues.append("Inappropriate guarantee claims detected")
        results["compliance_score"] = 65
    
    if any(term in model_output.lower() for term in ["ssn", "social security", "account number"]):
        compliance_issues.append("PII detected in output")
        results["safety_score"] = 45
    
    if "always" in model_output.lower() or "never" in model_output.lower():
        compliance_issues.append("Absolute statements may indicate bias")
        results["compliance_score"] = min(results["compliance_score"], 75)
    
    # Determine overall status
    failed_checks = []
    for metric, value in results.items():
        threshold = EVALUATION_METRICS[metric]["threshold"]
        if metric in ["hallucination_risk"] and value > threshold:
            failed_checks.append(f"{metric}: {value:.1f}% (threshold: <{threshold}%)")
        elif metric not in ["hallucination_risk"] and value < threshold:
            failed_checks.append(f"{metric}: {value:.1f}% (threshold: >{threshold}%)")
    
    overall_status = "PASSED" if not failed_checks and not compliance_issues else "FLAGGED"
    
    return {
        "status": overall_status,
        "scores": results,
        "compliance_issues": compliance_issues,
        "failed_checks": failed_checks,
        "evaluation_id": str(uuid.uuid4())[:8]
    }

def main():
    st.title("🛠️ GenAI Dev Core MVP Demo")
    st.markdown("*Unified GenAI Evaluation, Monitoring & Compliance Platform*")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Evaluation Framework",
        "📊 Monitoring Dashboard", 
        "🔧 Integration SDK",
        "📋 Audit Logging"
    ])
    
    with tab1:
        show_evaluation_framework()
    
    with tab2:
        show_monitoring_dashboard()
    
    with tab3:
        show_integration_sdk()
    
    with tab4:
        show_audit_logging()

def show_evaluation_framework():
    st.header("🔍 Evaluation Framework")
    st.markdown("*Accuracy scoring, safety checks, and financial compliance evaluation*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Evaluate Model Output")
        
        # Team and model selection
        team = st.selectbox("Team:", ["Budget Team", "Investment Team", "Lending Team", "Credit Team"])
        model = st.selectbox("Model Used:", ["GPT-4", "Claude-3.5", "GPT-3.5", "Internal Model"])
        
        # Input and output
        user_input = st.text_area("Original User Input:", 
                                 "What's the best way to invest $10,000 for retirement?")
        
        model_output = st.text_area("Model Output to Evaluate:", 
                                   "Investing $10,000 for retirement is guaranteed to make you rich if you put it all in stocks. This is completely risk-free and always works for everyone.",
                                   height=100)
        
        evaluation_type = st.selectbox("Evaluation Type:", ["Full Evaluation", "Safety Only", "Compliance Only"])
        
        if st.button("🔍 Run Evaluation", type="primary"):
            with st.spinner("Running evaluation framework..."):
                evaluation_results = evaluate_output(model_output, evaluation_type.lower())
            
            # Display results
            if evaluation_results["status"] == "PASSED":
                st.success(f"✅ EVALUATION PASSED (ID: {evaluation_results['evaluation_id']})")
            else:
                st.error(f"⚠️ EVALUATION FLAGGED (ID: {evaluation_results['evaluation_id']})")
            
            # Show detailed scores
            st.subheader("📊 Evaluation Scores")
            
            scores_data = []
            for metric, score in evaluation_results["scores"].items():
                threshold = EVALUATION_METRICS[metric]["threshold"]
                unit = EVALUATION_METRICS[metric]["unit"]
                
                if metric == "hallucination_risk":
                    status = "✅ Pass" if score <= threshold else "❌ Fail"
                    comparison = f"≤ {threshold}{unit}"
                else:
                    status = "✅ Pass" if score >= threshold else "❌ Fail"
                    comparison = f"≥ {threshold}{unit}"
                
                scores_data.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Score": f"{score:.1f}{unit}",
                    "Threshold": comparison,
                    "Status": status
                })
            
            scores_df = pd.DataFrame(scores_data)
            st.dataframe(scores_df, hide_index=True)
            
            # Show issues
            if evaluation_results["compliance_issues"]:
                st.subheader("🚨 Compliance Issues")
                for issue in evaluation_results["compliance_issues"]:
                    st.error(f"• {issue}")
            
            if evaluation_results["failed_checks"]:
                st.subheader("⚠️ Failed Metric Checks")
                for check in evaluation_results["failed_checks"]:
                    st.warning(f"• {check}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if evaluation_results["status"] == "FLAGGED":
                st.markdown("""
                **Suggested Improvements:**
                - Remove guarantee language and absolute claims
                - Add appropriate risk disclaimers
                - Include personalized risk assessment
                - Consider user's individual circumstances
                """)
            else:
                st.markdown("✅ Output meets all evaluation criteria and compliance requirements.")
    
    with col2:
        st.subheader("Active Compliance Rules")
        
        st.markdown("**Financial Compliance Checks:**")
        for rule in FINANCIAL_COMPLIANCE_RULES:
            st.markdown(f"**{rule['rule']}**")
            st.caption(rule['description'])
            st.markdown("---")
        
        # Evaluation metrics reference
        st.subheader("Evaluation Thresholds")
        
        thresholds_data = []
        for metric, config in EVALUATION_METRICS.items():
            thresholds_data.append({
                "Metric": metric.replace("_", " ").title(),
                "Threshold": f"{config['threshold']}{config['unit']}",
                "Type": "Maximum" if metric == "hallucination_risk" or metric == "response_latency" else "Minimum"
            })
        
        thresholds_df = pd.DataFrame(thresholds_data)
        st.dataframe(thresholds_df, hide_index=True)

def show_monitoring_dashboard():
    st.header("📊 Monitoring Dashboard")
    st.markdown("*Real-time monitoring of latency, failure rates, and model drift*")
    
    # Generate monitoring data
    monitoring_df = generate_monitoring_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = monitoring_df.groupby('team').last().reset_index()
    total_requests = latest_data['requests'].sum()
    avg_failure_rate = latest_data['failure_rate'].mean()
    avg_latency = latest_data['avg_latency'].mean()
    total_violations = latest_data['compliance_violations'].sum()
    
    col1.metric("Total Requests (Last Hour)", f"{total_requests:,}", "+12%")
    col2.metric("Avg Failure Rate", f"{avg_failure_rate:.1f}%", "-0.3%")
    col3.metric("Avg Latency", f"{avg_latency:.0f}ms", "+15ms")
    col4.metric("Compliance Violations", total_violations, "-2")
    
    # Charts
    st.subheader("Performance Trends (Last 24 Hours)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Latency over time
        hourly_latency = monitoring_df.groupby('timestamp')['avg_latency'].mean().reset_index()
        fig = px.line(hourly_latency, x='timestamp', y='avg_latency',
                     title="Average Response Latency")
        fig.add_hline(y=2000, line_dash="dash", line_color="red", 
                     annotation_text="SLA: 2000ms")
        fig.update_layout(xaxis_title="Time", yaxis_title="Latency (ms)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Failure rate by team
        team_failures = monitoring_df.groupby('team')['failure_rate'].mean().reset_index()
        fig = px.bar(team_failures, x='team', y='failure_rate',
                    title="Average Failure Rate by Team")
        fig.update_layout(xaxis_title="Team", yaxis_title="Failure Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model drift detection
    st.subheader("Model Performance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy scores over time
        hourly_accuracy = monitoring_df.groupby('timestamp')['accuracy_score'].mean().reset_index()
        fig = px.line(hourly_accuracy, x='timestamp', y='accuracy_score',
                     title="Model Accuracy Trend")
        fig.add_hline(y=85, line_dash="dash", line_color="red", 
                     annotation_text="Threshold: 85%")
        fig.update_layout(xaxis_title="Time", yaxis_title="Accuracy Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Safety scores
        hourly_safety = monitoring_df.groupby('timestamp')['safety_score'].mean().reset_index()
        fig = px.line(hourly_safety, x='timestamp', y='safety_score',
                     title="Safety Score Trend")
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Threshold: 90%")
        fig.update_layout(xaxis_title="Time", yaxis_title="Safety Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts section
    st.subheader("🚨 Active Alerts")
    
    alerts_data = [
        {"Alert": "High Latency", "Team": "Investment Team", "Severity": "Warning", "Duration": "15 min"},
        {"Alert": "Compliance Violations", "Team": "Lending Team", "Severity": "Critical", "Duration": "5 min"},
        {"Alert": "Model Drift Detected", "Team": "Budget Team", "Severity": "Info", "Duration": "1 hour"}
    ]
    
    alerts_df = pd.DataFrame(alerts_data)
    
    # Color code severity
    def color_severity(val):
        if val == "Critical":
            return "background-color: #ffebee"
        elif val == "Warning":
            return "background-color: #fff3e0" 
        else:
            return "background-color: #e8f5e8"
    
    st.dataframe(alerts_df.style.applymap(color_severity, subset=['Severity']), hide_index=True)

def show_integration_sdk():
    st.header("🔧 Integration SDK")
    st.markdown("*Simple API and CLI tools for sending model outputs for evaluation*")
    
    tab1, tab2, tab3 = st.tabs(["Python SDK", "REST API", "CLI Tool"])
    
    with tab1:
        st.subheader("Python SDK Integration")
        
        # Installation
        st.markdown("**Installation:**")
        st.code("pip install genai-dev-core", language="bash")
        
        # Basic usage
        st.markdown("**Basic Usage:**")
        st.code("""
from genai_dev_core import EvaluationClient

# Initialize client
client = EvaluationClient(
    api_key="your_api_key",
    team="budget_team"
)

# Evaluate model output
result = client.evaluate(
    model_output="Your budget shows 65% going to housing, which exceeds recommended limits.",
    user_input="Analyze my monthly spending breakdown",
    model_name="gpt-4",
    evaluation_type="full"  # or "safety_only", "compliance_only"
)

# Check results
if result.status == "passed":
    print("✅ Output approved for production")
    print(f"Accuracy: {result.scores.accuracy_score}%")
else:
    print("⚠️ Issues detected:")
    for issue in result.compliance_issues:
        print(f"  - {issue}")
        """, language="python")
        
        # Advanced features
        st.markdown("**Advanced Features:**")
        st.code("""
# Batch evaluation
results = client.evaluate_batch([
    {"output": "Investment advice 1", "input": "Question 1"},
    {"output": "Investment advice 2", "input": "Question 2"}
])

# Custom evaluation criteria
result = client.evaluate(
    model_output="...",
    custom_rules=["strict_financial_compliance", "high_accuracy_mode"],
    metadata={"feature": "portfolio_advisor", "version": "v2.1"}
)

# Monitor real-time
client.start_monitoring(
    callback=lambda result: handle_evaluation(result),
    alert_on=["compliance_violations", "accuracy_drop"]
)
        """, language="python")
    
    with tab2:
        st.subheader("REST API Documentation")
        
        # API endpoint
        st.markdown("**Base URL:** `https://genai-dev-core.internal.company.com/api/v1`")
        
        # Authentication
        st.markdown("**Authentication:**")
        st.code("""
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     -H "Content-Type: application/json"
        """, language="bash")
        
        # Evaluation endpoint
        st.markdown("**POST /evaluate**")
        st.code("""
{
  "team": "budget_team",
  "model_name": "gpt-4",
  "user_input": "How should I budget $5000 monthly income?",
  "model_output": "Allocate 50% to needs, 30% to wants, 20% to savings and debt repayment.",
  "evaluation_type": "full",
  "metadata": {
    "feature": "budget_advisor",
    "version": "v1.2"
  }
}
        """, language="json")
        
        st.markdown("**Response:**")
        st.code("""
{
  "evaluation_id": "eval_abc123",
  "status": "passed",
  "scores": {
    "accuracy_score": 92.3,
    "safety_score": 96.1,
    "compliance_score": 94.8,
    "hallucination_risk": 3.2,
    "response_latency": 1250
  },
  "compliance_issues": [],
  "failed_checks": [],
  "timestamp": "2025-08-08T15:30:00Z"
}
        """, language="json")
        
        # Monitoring endpoint
        st.markdown("**GET /monitoring/{team}**")
        st.markdown("Returns real-time monitoring data for the specified team.")
        
        # Batch endpoint
        st.markdown("**POST /evaluate/batch**")
        st.markdown("Evaluate multiple outputs in a single request (up to 100 items).")
    
    with tab3:
        st.subheader("CLI Tool")
        
        # Installation
        st.markdown("**Installation:**")
        st.code("npm install -g @company/genai-dev-core-cli", language="bash")
        
        # Setup
        st.markdown("**Initial Setup:**")
        st.code("""
# Configure authentication
genai-core config set-token YOUR_API_KEY
genai-core config set-team budget_team

# Verify connection
genai-core status
        """, language="bash")
        
        # Basic commands
        st.markdown("**Basic Commands:**")
        st.code("""
# Evaluate single output
genai-core evaluate \\
  --input "What's the best savings account?" \\
  --output "High-yield savings accounts typically offer 4-5% APY" \\
  --model gpt-4

# Evaluate from file
genai-core evaluate --file model_outputs.json

# Monitor team performance
genai-core monitor --team budget_team --realtime

# Get evaluation history
genai-core history --last 24h --format table
        """, language="bash")
        
        # Integration with CI/CD
        st.markdown("**CI/CD Integration:**")
        st.code("""
# In your deployment pipeline
- name: Evaluate Model Outputs
  run: |
    genai-core evaluate --file test_outputs.json --fail-on-violations
    if [ $? -eq 0 ]; then
      echo "✅ All evaluations passed"
    else
      echo "❌ Evaluation failures detected"
      exit 1
    fi
        """, language="yaml")
    
    # Interactive demo
    st.subheader("🧪 Try the SDK")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Test API Call:**")
        demo_team = st.selectbox("Team:", ["Budget Team", "Investment Team", "Lending Team"])
        demo_model = st.selectbox("Model:", ["GPT-4", "Claude-3.5", "GPT-3.5"])
        demo_input = st.text_area("User Input:", "Should I refinance my mortgage?")
        demo_output = st.text_area("Model Output:", "Refinancing could save you money if rates have dropped since your original loan.")
        
        if st.button("🚀 Test Evaluation API"):
            with st.spinner("Calling evaluation API..."):
                time.sleep(1)
            
            st.success("✅ API call successful!")
            st.code(f"""
Response:
{{
  "evaluation_id": "eval_{random.randint(100000, 999999)}",
  "status": "passed",
  "scores": {{
    "accuracy_score": {random.uniform(85, 95):.1f},
    "safety_score": {random.uniform(90, 98):.1f},
    "compliance_score": {random.uniform(88, 96):.1f}
  }},
  "processing_time": "{random.randint(800, 1500)}ms"
}}
            """, language="json")
    
    with col2:
        st.markdown("**Generated SDK Code:**")
        generated_code = f"""
from genai_dev_core import EvaluationClient

client = EvaluationClient(team="{demo_team.lower().replace(' ', '_')}")

result = client.evaluate(
    model_output="{demo_output[:50]}...",
    user_input="{demo_input[:30]}...",
    model_name="{demo_model.lower()}"
)

print(f"Status: {{result.status}}")
print(f"Accuracy: {{result.scores.accuracy_score}}%")
        """
        st.code(generated_code, language="python")

def show_audit_logging():
    st.header("📋 Audit Logging")
    st.markdown("*Comprehensive logging of evaluations and decisions for compliance review*")
    
    # Search and filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        date_filter = st.date_input("Date Range Start:", datetime.now().date() - timedelta(days=7))
    with col2:
        team_filter = st.selectbox("Filter by Team:", ["All Teams", "Budget Team", "Investment Team", "Lending Team"])
    with col3:
        status_filter = st.selectbox("Filter by Status:", ["All", "Passed", "Flagged", "Failed"])
    with col4:
        model_filter = st.selectbox("Filter by Model:", ["All Models", "GPT-4", "Claude-3.5", "GPT-3.5"])
    
    # Generate audit log data
    audit_logs = []
    for i in range(50):
        log_entry = {
            "evaluation_id": f"eval_{random.randint(100000, 999999)}",
            "timestamp": datetime.now() - timedelta(hours=random.randint(0, 168)),
            "team": random.choice(["Budget Team", "Investment Team", "Lending Team", "Credit Team"]),
            "model": random.choice(["GPT-4", "Claude-3.5", "GPT-3.5"]),
            "status": random.choice(["Passed", "Flagged", "Failed"]),
            "accuracy_score": random.uniform(70, 98),
            "compliance_score": random.uniform(75, 99),
            "issues_count": random.randint(0, 3),
            "user_input": f"Sample user input {i+1}...",
            "model_output": f"Sample model output {i+1}...",
        }
        audit_logs.append(log_entry)
    
    audit_df = pd.DataFrame(audit_logs)
    
    # Apply filters
    if team_filter != "All Teams":
        audit_df = audit_df[audit_df['team'] == team_filter]
    if status_filter != "All":
        audit_df = audit_df[audit_df['status'] == status_filter]
    if model_filter != "All Models":
        audit_df = audit_df[audit_df['model'] == model_filter]
    
    # Summary stats
    st.subheader("📊 Audit Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluations", len(audit_df))
    col2.metric("Pass Rate", f"{(audit_df['status'] == 'Passed').mean() * 100:.1f}%")
    col3.metric("Avg Accuracy", f"{audit_df['accuracy_score'].mean():.1f}%")
    col4.metric("Compliance Issues", audit_df['issues_count'].sum())
    
    # Detailed logs table
    st.subheader("🔍 Detailed Audit Logs")
    
    # Select columns to display
    display_columns = st.multiselect(
        "Select columns to display:",
        ["evaluation_id", "timestamp", "team", "model", "status", "accuracy_score", "compliance_score", "issues_count"],
        default=["evaluation_id", "timestamp", "team", "status", "accuracy_score", "compliance_score"]
    )
    
    if display_columns:
        # Format timestamp for display
        display_df = audit_df[display_columns].copy()
        if 'timestamp' in display_columns:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        if 'accuracy_score' in display_columns:
            display_df['accuracy_score'] = display_df['accuracy_score'].round(1)
        if 'compliance_score' in display_columns:
            display_df['compliance_score'] = display_df['compliance_score'].round(1)
        
        st.dataframe(display_df, hide_index=True, height=400)
    
    # Export functionality
    st.subheader("📤 Export Audit Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            st.success("✅ CSV export initiated - check your downloads")
    
    with col2:
        if st.button("📋 Generate Compliance Report"):
            st.success("✅ Compliance report generated")
    
    with col3:
        if st.button("🔍 Detailed Investigation"):
            st.info("🔍 Opening detailed investigation dashboard...")
    
    # Compliance violations detail
    if status_filter == "Flagged" or status_filter == "All":
        st.subheader("⚠️ Recent Compliance Violations")
        
        violations_data = [
            {
                "Evaluation ID": "eval_789123",
                "Team": "Investment Team", 
                "Issue": "Risk disclosure missing",
                "Severity": "Medium",
                "Timestamp": "2025-08-08 14:30",
                "Resolved": "No"
            },
            {
                "Evaluation ID": "eval_456789",
                "Team": "Lending Team",
                "Issue": "PII detected in output",
                "Severity": "High", 
                "Timestamp": "2025-08-08 13:15",
                "Resolved": "Yes"
            },
            {
                "Evaluation ID": "eval_321654",
                "Team": "Budget Team",
                "Issue": "Absolute guarantee claims",
                "Severity": "Medium",
                "Timestamp": "2025-08-08 12:45",
                "Resolved": "No"
            }
        ]
        
        violations_df = pd.DataFrame(violations_data)
        st.dataframe(violations_df, hide_index=True)

if __name__ == "__main__":
    main()
