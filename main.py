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

# Mock data for the new features
FINANCIAL_USE_CASES = {
    "Budget Analysis": {
        "description": "Analyze spending patterns and provide improvement suggestions",
        "optimal_model": "GPT-4",
        "success_rate": 94.2,
        "avg_cost": 0.034
    },
    "Investment Advice": {
        "description": "Generate personalized investment recommendations",
        "optimal_model": "Claude-3.5",
        "success_rate": 91.8,
        "avg_cost": 0.042
    },
    "Risk Assessment": {
        "description": "Evaluate financial risk tolerance and profile",
        "optimal_model": "GPT-4",
        "success_rate": 89.6,
        "avg_cost": 0.038
    },
    "Loan Recommendations": {
        "description": "Match users with appropriate loan products",
        "optimal_model": "Claude-3.5",
        "success_rate": 87.3,
        "avg_cost": 0.029
    }
}

PROMPT_VERSIONS = {
    "budget_analysis_v1": {
        "prompt": "Analyze this budget and find problems.",
        "success_rate": 67.2,
        "status": "deprecated"
    },
    "budget_analysis_v2": {
        "prompt": "Analyze the user's budget focusing on: 1) spending vs income ratio 2) category allocation vs recommended limits 3) specific improvement opportunities",
        "success_rate": 84.7,
        "status": "active"
    },
    "budget_analysis_v3": {
        "prompt": "You are a certified financial planner. Analyze this budget data and provide specific, actionable recommendations. Consider: spending ratios, emergency fund status, and optimization opportunities. Be encouraging but realistic.",
        "success_rate": 94.2,
        "status": "optimized"
    }
}

COMPLIANCE_SCENARIOS = [
    {
        "text": "This investment is guaranteed to double your money with zero risk.",
        "issues": ["Misleading guarantee claims", "Risk misrepresentation"],
        "context": "High-risk investment advice without proper disclaimers",
        "suggestion": "Add risk disclaimers and remove guarantee language"
    },
    {
        "text": "Based on your income of $85,000, I recommend investing in growth stocks.",
        "issues": [],
        "context": "Appropriate advice with income context",
        "suggestion": "Advice is compliant and well-contextualized"
    },
    {
        "text": "Young people should always choose aggressive investments since they have time.",
        "issues": ["Age-based generalization", "Lacks individual context"],
        "context": "Generic advice without personal risk assessment",
        "suggestion": "Include individual risk tolerance assessment"
    }
]

@st.cache_data
def generate_optimization_data():
    """Generate mock data showing AI optimization improvements over time"""
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    
    # Model routing optimization data
    routing_data = []
    base_accuracy = 78
    for i, date in enumerate(dates):
        # Simulate learning curve
        accuracy_improvement = min(16, i * 0.6)  # Max 16% improvement
        current_accuracy = base_accuracy + accuracy_improvement
        
        routing_data.append({
            "date": date,
            "accuracy": current_accuracy,
            "cost_savings": min(30, i * 1.2),  # Max 30% cost savings
            "optimal_routes": min(95, 65 + i * 1.0)  # Routing accuracy
        })
    
    # Prompt optimization data
    prompt_data = []
    for use_case, data in FINANCIAL_USE_CASES.items():
        for i in range(7):  # Weekly data points
            week_date = datetime.now() - timedelta(weeks=i)
            base_rate = data["success_rate"] - 15  # Starting point
            improvement = min(15, (7-i) * 2.5)  # Gradual improvement
            
            prompt_data.append({
                "date": week_date,
                "use_case": use_case,
                "success_rate": base_rate + improvement,
                "tests_run": random.randint(20, 100),
                "version": f"v{3-min(2, i//2)}"  # Version progression
            })
    
    return pd.DataFrame(routing_data), pd.DataFrame(prompt_data)

def simulate_smart_routing(use_case, user_context):
    """Simulate AI model routing decision"""
    if use_case not in FINANCIAL_USE_CASES:
        return "GPT-3.5", "No optimization data available"
    
    data = FINANCIAL_USE_CASES[use_case]
    
    # Simulate decision factors
    factors = {
        "Historical Performance": f"{data['success_rate']}% success rate",
        "Cost Efficiency": f"${data['avg_cost']:.3f} per request",
        "User Context": "High-value customer" if "investment" in user_context.lower() else "Standard user",
        "Model Load": "Normal" if random.random() > 0.3 else "High load - fallback considered"
    }
    
    return data["optimal_model"], factors

def simulate_prompt_generation(description):
    """Simulate AI-generated prompt from description"""
    base_templates = {
        "budget": "You are a financial advisor analyzing a user's budget. {description}. Provide specific, actionable recommendations with clear reasoning.",
        "investment": "As an investment advisor, {description}. Consider the user's risk profile and provide personalized guidance with appropriate disclaimers.",
        "loan": "You are a lending specialist helping users {description}. Evaluate their financial profile and recommend suitable products with clear terms.",
        "risk": "As a risk assessment expert, {description}. Analyze all relevant factors and provide a balanced evaluation."
    }
    
    # Simple keyword matching for demo
    for key, template in base_templates.items():
        if key in description.lower():
            return template.format(description=description.lower())
    
    return f"Based on your request to '{description}', analyze the user's financial situation and provide personalized, compliant advice with clear reasoning and appropriate disclaimers."

def main():
    st.title("🧠 FinPlan GenAI Platform MVP Demo")
    st.markdown("*AI-Native Platform: Smart Infrastructure + Intelligent Optimization*")
    
    # Create tabs for the three main features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔀 AI-Optimized Gateway", 
        "✨ Self-Improving Prompts", 
        "🛡️ Intelligent Compliance",
        "📊 Platform Analytics"
    ])
    
    with tab1:
        show_ai_gateway()
    
    with tab2:
        show_prompt_engine()
    
    with tab3:
        show_intelligent_compliance()
    
    with tab4:
        show_platform_analytics()

def show_ai_gateway():
    st.header("🔀 AI-Optimized Gateway")
    st.markdown("*Intelligent model routing that learns optimal performance patterns*")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Smart Routing Demo")
        
        # Use case selection
        use_case = st.selectbox("Financial Use Case:", list(FINANCIAL_USE_CASES.keys()))
        
        # User context
        user_context = st.text_area("User Context:", 
                                   "High-income professional, age 35, moderate risk tolerance, existing investment portfolio")
        
        # Request input
        request_text = st.text_area("Request Content:", 
                                   "Analyze my monthly spending of $4,500 and suggest optimizations")
        
        if st.button("Send Request with Smart Routing", type="primary"):
            with st.spinner("🧠 AI analyzing optimal routing..."):
                time.sleep(1.5)
                
                # Simulate routing decision
                selected_model, routing_factors = simulate_smart_routing(use_case, user_context)
                
                st.success(f"✅ Request routed to {selected_model}")
                
                # Show routing decision factors
                st.markdown("**🎯 Routing Decision Factors:**")
                for factor, value in routing_factors.items():
                    st.write(f"• **{factor}:** {value}")
                
                # Mock response
                st.markdown("**💬 AI Response:**")
                if "budget" in use_case.lower():
                    response = """**Budget Analysis Results:**

Your monthly spending of $4,500 shows several optimization opportunities:

**Spending Breakdown Analysis:**
- Housing costs appear to be within reasonable range
- Food and dining: Consider meal planning to reduce by 10-15%
- Entertainment: Current level is sustainable for your income

**Specific Recommendations:**
1. **Emergency Fund:** Prioritize building 3-6 months of expenses
2. **Automated Savings:** Set up automatic transfer of $800/month
3. **Category Optimization:** Reduce discretionary spending by $200/month

**Next Steps:**
- Review subscription services for cancellation opportunities
- Consider the 50/30/20 budgeting rule for better allocation"""
                else:
                    response = f"AI-generated response for {use_case} would appear here with contextual recommendations based on the user profile and request."
                
                st.markdown(response)
    
    with col2:
        st.subheader("Routing Performance")
        
        # Performance metrics
        if use_case in FINANCIAL_USE_CASES:
            data = FINANCIAL_USE_CASES[use_case]
            st.metric("Success Rate", f"{data['success_rate']:.1f}%", "+12.3%")
            st.metric("Avg Cost", f"${data['avg_cost']:.3f}", "-18%")
            st.metric("Optimal Model", data['optimal_model'])
        
        # Learning curve visualization
        st.markdown("**📈 AI Learning Progress**")
        routing_data, _ = generate_optimization_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=routing_data['date'], 
            y=routing_data['accuracy'],
            mode='lines+markers',
            name='Routing Accuracy',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title="Model Routing Accuracy Over Time",
            xaxis_title="Date",
            yaxis_title="Accuracy (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost savings
        st.metric("Cost Savings", f"{routing_data['cost_savings'].iloc[-1]:.1f}%", "+2.1%")

def show_prompt_engine():
    st.header("✨ Self-Improving Prompt Engine")
    st.markdown("*AI-generated prompts that continuously optimize based on performance*")
    
    tab1, tab2, tab3 = st.tabs(["Generate New Prompt", "Prompt Evolution", "Performance Analytics"])
    
    with tab1:
        st.subheader("🎯 Generate Optimized Prompt")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Describe what you need:**")
            description = st.text_area("Natural Language Description:", 
                                     "help users understand if they're overspending on housing")
            
            target_audience = st.selectbox("Target Audience:", 
                                         ["First-time homebuyers", "Young professionals", "Families", "Retirees"])
            
            tone = st.selectbox("Desired Tone:", 
                              ["Professional", "Friendly", "Educational", "Encouraging"])
            
            if st.button("🚀 Generate Optimized Prompt"):
                with st.spinner("🧠 AI crafting optimal prompt..."):
                    time.sleep(2)
                
                generated_prompt = simulate_prompt_generation(description)
                
                st.success("✅ Prompt generated and optimized!")
                st.markdown("**Generated Prompt:**")
                st.code(generated_prompt, language="text")
                
                # Show optimization details
                st.markdown("**🎯 Optimization Applied:**")
                st.write("• Added financial advisor persona for authority")
                st.write("• Included disclaimer requirements for compliance")
                st.write("• Structured output for better user experience")
                st.write("• Incorporated tone preferences")
        
        with col2:
            st.markdown("**🎲 Test Generated Prompt:**")
            test_data = st.text_area("Sample User Data:", 
                                    '{"monthly_income": 6500, "housing_cost": 2800, "other_expenses": 2200}')
            
            if st.button("Test Prompt Performance"):
                st.info("📊 Simulating prompt performance...")
                
                # Mock performance results
                performance_df = pd.DataFrame([
                    {"Metric": "Relevance Score", "Value": "92.4%"},
                    {"Metric": "User Satisfaction", "Value": "4.7/5.0"},
                    {"Metric": "Compliance Check", "Value": "✅ Passed"},
                    {"Metric": "Response Time", "Value": "1.2s"},
                    {"Metric": "Cost", "Value": "$0.034"}
                ])
                st.dataframe(performance_df, hide_index=True)
    
    with tab2:
        st.subheader("📈 Prompt Evolution History")
        
        selected_case = st.selectbox("View Evolution for:", ["Budget Analysis", "Investment Advice", "Risk Assessment"])
        
        # Show version progression
        st.markdown(f"**Evolution of {selected_case} Prompts:**")
        
        for version, data in PROMPT_VERSIONS.items():
            status_emoji = {"deprecated": "🔴", "active": "🟡", "optimized": "🟢"}
            status = data["status"]
            
            st.markdown(f"**{version.upper()}** {status_emoji[status]} *{status}*")
            st.code(data["prompt"])
            st.write(f"Success Rate: {data['success_rate']:.1f}%")
            st.markdown("---")
    
    with tab3:
        st.subheader("📊 Prompt Performance Analytics")
        
        # Generate and display prompt performance data
        _, prompt_data = generate_optimization_data()
        
        # Performance by use case
        col1, col2 = st.columns(2)
        
        with col1:
            latest_performance = prompt_data.groupby('use_case')['success_rate'].last().reset_index()
            fig = px.bar(latest_performance, x='use_case', y='success_rate',
                        title="Current Success Rates by Use Case")
            fig.update_layout(xaxis_title="Use Case", yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Improvement over time
            budget_data = prompt_data[prompt_data['use_case'] == 'Budget Analysis']
            fig = px.line(budget_data, x='date', y='success_rate',
                         title="Budget Analysis Prompt Improvement")
            fig.update_layout(xaxis_title="Date", yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

def show_intelligent_compliance():
    st.header("🛡️ Intelligent Compliance Guardian")
    st.markdown("*Context-aware compliance that understands financial advice nuances*")
    
    tab1, tab2, tab3 = st.tabs(["Live Compliance Check", "Compliance Learning", "Audit Dashboard"])
    
    with tab1:
        st.subheader("🔍 Context-Aware Compliance Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Input for compliance checking
            advice_text = st.text_area("Financial Advice to Check:", 
                                     "Based on your age and income, you should put all your money in cryptocurrency since you're young and can recover from losses.",
                                     height=100)
            
            user_profile = st.text_area("User Context:", 
                                      '{"age": 25, "income": 45000, "risk_tolerance": "moderate", "investment_experience": "beginner"}',
                                      height=80)
            
            advice_type = st.selectbox("Advice Category:", 
                                     ["Investment Recommendation", "Budget Advice", "Loan Guidance", "Risk Assessment"])
            
            if st.button("🔍 Run Intelligent Compliance Check", type="primary"):
                with st.spinner("🧠 AI analyzing compliance and context..."):
                    time.sleep(2)
                
                # Simulate intelligent compliance analysis
                st.markdown("**🚨 Compliance Analysis Results:**")
                
                # Risk assessment
                st.error("**HIGH RISK DETECTED**")
                
                issues_found = [
                    "Inappropriate risk recommendation for user profile",
                    "Lack of diversification advice",
                    "Missing risk disclaimers",
                    "Age-based stereotyping in investment advice"
                ]
                
                st.markdown("**Issues Identified:**")
                for issue in issues_found:
                    st.write(f"🔴 {issue}")
                
                # Context-aware suggestions
                st.markdown("**💡 Intelligent Suggestions:**")
                suggestions = [
                    "Consider user's 'moderate' risk tolerance instead of assuming high risk appetite",
                    "Recommend diversified portfolio appropriate for beginner investor",
                    "Add standard investment risk disclaimers",
                    "Provide education about cryptocurrency risks before any allocation recommendation"
                ]
                
                for suggestion in suggestions:
                    st.write(f"✅ {suggestion}")
                
                # Improved version
                st.markdown("**📝 Suggested Compliant Version:**")
                improved_text = """Based on your moderate risk tolerance and beginner investment experience, I recommend starting with a diversified portfolio including low-cost index funds. While cryptocurrency can be part of a portfolio, it should typically represent no more than 5-10% of total investments due to high volatility. 

Consider beginning with:
- 60% stock index funds
- 30% bond index funds  
- 10% alternative investments (which could include some cryptocurrency)

*Disclaimer: All investments carry risk of loss. Past performance does not guarantee future results. Consider consulting with a financial advisor for personalized advice.*"""
                
                st.success("**Improved Compliant Advice:**")
                st.write(improved_text)
        
        with col2:
            st.subheader("Compliance Score")
            
            # Mock compliance scoring
            st.metric("Overall Score", "23/100", "🔴 High Risk")
            st.metric("Risk Assessment", "12/25", "Poor")
            st.metric("Regulatory Compliance", "8/25", "Poor") 
            st.metric("Context Appropriateness", "3/25", "Poor")
            st.metric("Disclaimer Coverage", "0/25", "Missing")
            
            st.markdown("**🎯 Compliance Factors:**")
            factors_df = pd.DataFrame([
                {"Factor": "User Risk Match", "Score": "Low", "Weight": "25%"},
                {"Factor": "Age Appropriateness", "Score": "Low", "Weight": "20%"},
                {"Factor": "Experience Level", "Score": "Low", "Weight": "20%"},
                {"Factor": "Diversification", "Score": "Low", "Weight": "15%"},
                {"Factor": "Disclaimers", "Score": "Missing", "Weight": "20%"}
            ])
            st.dataframe(factors_df, hide_index=True)
    
    with tab2:
        st.subheader("🧠 Compliance Learning System")
        
        st.markdown("**Review and Improve Detection:**")
        
        # Show example scenarios for training
        scenario_idx = st.selectbox("Review Compliance Scenario:", range(len(COMPLIANCE_SCENARIOS)))
        scenario = COMPLIANCE_SCENARIOS[scenario_idx]
        
        st.markdown("**Financial Advice:**")
        st.code(scenario["text"])
        
        st.markdown("**AI Assessment:**")
        if scenario["issues"]:
            st.error(f"Issues Detected: {', '.join(scenario['issues'])}")
        else:
            st.success("No compliance issues detected")
        
        st.markdown(f"**Context Understanding:** {scenario['context']}")
        st.markdown(f"**AI Suggestion:** {scenario['suggestion']}")
        
        # Feedback mechanism
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Correct Assessment"):
                st.success("Thank you! AI learning updated.")
        with col2:
            if st.button("❌ Incorrect Assessment"):
                st.warning("Feedback recorded for AI improvement.")
        with col3:
            if st.button("🔄 Partially Correct"):
                st.info("Nuanced feedback saved for training.")
    
    with tab3:
        st.subheader("📊 Compliance Audit Dashboard")
        
        # Compliance metrics over time
        dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
        compliance_data = []
        
        for date in dates:
            compliance_data.append({
                "date": date,
                "requests_scanned": random.randint(800, 1200),
                "issues_detected": random.randint(15, 45),
                "high_risk_blocked": random.randint(2, 8),
                "accuracy_score": random.uniform(92, 98)
            })
        
        compliance_df = pd.DataFrame(compliance_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Requests Scanned Today", "1,147", "+23")
        col2.metric("Issues Detected", "28", "-5")
        col3.metric("High-Risk Blocked", "6", "+2")
        col4.metric("Detection Accuracy", "96.2%", "+1.1%")
        
        # Trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(compliance_df, x='date', y='accuracy_score',
                         title="Compliance Detection Accuracy Trend")
            fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(compliance_df, x='date', y='issues_detected',
                         title="Daily Compliance Issues Detected")
            fig.update_layout(yaxis_title="Issues Count", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)

def show_platform_analytics():
    st.header("📊 Platform Analytics")
    st.markdown("*Comprehensive view of AI optimization and platform performance*")
    
    # Generate comprehensive analytics data
    routing_data, prompt_data = generate_optimization_data()
    
    # Overall platform metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Teams", "6", "+2")
    col2.metric("AI Performance Gain", "+18.3%", "+2.1%")
    col3.metric("Cost Optimization", "24.7%", "+1.8%")
    col4.metric("Developer Satisfaction", "4.6/5.0", "+0.3")
    
    # Detailed analytics tabs
    tab1, tab2, tab3 = st.tabs(["🔀 Routing Intelligence", "✨ Prompt Evolution", "🛡️ Compliance Trends"])
    
    with tab1:
        st.subheader("AI Model Routing Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Routing accuracy over time
            fig = px.line(routing_data, x='date', y='accuracy',
                         title="Model Routing Accuracy Improvement")
            fig.add_hline(y=85, line_dash="dash", line_color="red", 
                         annotation_text="Target: 85%")
            fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost savings progression
            fig = px.line(routing_data, x='date', y='cost_savings',
                         title="Cumulative Cost Savings")
            fig.update_layout(yaxis_title="Cost Savings (%)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model usage distribution
        st.subheader("Model Usage Distribution")
        model_usage = pd.DataFrame([
            {"Model": "GPT-4", "Usage": 45, "Avg_Cost": 0.042, "Success_Rate": 92.1},
            {"Model": "Claude-3.5", "Usage": 35, "Avg_Cost": 0.038, "Success_Rate": 89.7},
            {"Model": "GPT-3.5", "Usage": 20, "Avg_Cost": 0.018, "Success_Rate": 84.3}
        ])
        
        fig = px.pie(model_usage, values='Usage', names='Model',
                    title="Model Usage by Request Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Prompt Performance Evolution")
        
        # Success rate improvements by use case
        latest_prompt_performance = prompt_data.groupby('use_case').agg({
            'success_rate': ['first', 'last'],
            'tests_run': 'sum'
        }).round(1)
        
        latest_prompt_performance.columns = ['Initial_Rate', 'Current_Rate', 'Total_Tests']
        latest_prompt_performance['Improvement'] = (
            latest_prompt_performance['Current_Rate'] - latest_prompt_performance['Initial_Rate']
        ).round(1)
        latest_prompt_performance = latest_prompt_performance.reset_index()
        
        st.dataframe(latest_prompt_performance, hide_index=True)
        
        # Prompt testing volume over time
        weekly_tests = prompt_data.groupby('date')['tests_run'].sum().reset_index()
        fig = px.bar(weekly_tests, x='date', y='tests_run',
                    title="Weekly Prompt A/B Tests Conducted")
        fig.update_layout(xaxis_title="Date", yaxis_title="Tests Conducted")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Compliance Intelligence Trends")
        
        # Mock compliance trend data
        compliance_trends = pd.DataFrame([
            {"Week": f"Week {i}", "Detection_Accuracy": 88 + i*1.2, "False_Positives": max(12 - i*0.8, 3)} 
            for i in range(8)
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(compliance_trends, x='Week', y='Detection_Accuracy',
                         title="Compliance Detection Accuracy Improvement")
            fig.update_layout(yaxis_title="Accuracy (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(compliance_trends, x='Week', y='False_Positives',
                         title="False Positive Rate Reduction")
            fig.update_layout(yaxis_title="False Positives (%)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
