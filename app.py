import streamlit as st
import pandas as pd
import numpy as np
# Visualization imports removed per user request
from utils_fixed import LBVPredictor, convert_units, get_input_ranges, format_prediction_result, get_model_info, initialize_model
import os

# Page configuration
st.set_page_config(
    page_title="🔥 Laminar Burning Velocity Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern teal/cyan theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .input-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(6, 182, 212, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .result-container {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(6, 182, 212, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-title {
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #06b6d4;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d9488 0%, #0ea5e9 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }
    
    .stSelectbox > div > div > div {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1e293b;
        border: 1px solid #475569;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state for prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Initialize model
    if not initialize_model():
        st.error("Failed to initialize model. Please check the data file.")
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔥 Laminar Burning Velocity Prediction</h1>
        <p>Predict the Laminar Burning Velocity (LBV) for different hydrocarbons based on temperature, equivalence ratio, and pressure.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    try:
        predictor = LBVPredictor()
        hydrocarbon_options = predictor.get_hydrocarbon_options()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sidebar with menu layout
    with st.sidebar:
        # Menu header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">📋 Menu</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # History toggle switch
        if 'show_history' not in st.session_state:
            st.session_state.show_history = True
            
        show_history = st.toggle("Show Prediction History", value=st.session_state.show_history)
        st.session_state.show_history = show_history
        
        st.markdown("---")
        
        # Model info section
        st.markdown("**Model Information**")
        st.markdown(f"- Algorithm: Random Forest")
        if predictor.data is not None:
            st.markdown(f"- Dataset: {len(predictor.data)} samples")
        st.markdown(f"- Hydrocarbons: {len(hydrocarbon_options)}")
        
        st.markdown("---")
        
        # Model Performance Metrics
        with st.expander("📊 Model Performance", expanded=False):
            metrics = predictor.get_model_metrics()
            if metrics:
                st.metric("R² Score", f"{metrics['r2']:.3f}")
                st.metric("RMSE", f"{metrics['rmse']:.2f} cm/s")
                st.metric("MAE", f"{metrics['mae']:.2f} cm/s")
                st.metric("±2 cm/s Accuracy", f"{metrics['accuracy_percent']:.1f}%")
                
                st.markdown("**Best Parameters:**")
                for param, value in metrics['best_params'].items():
                    st.write(f"• {param}: {value}")
            else:
                st.write("**Algorithm:** Random Forest Regression")
                st.write("**Dataset Size:** 5000+ data points")
                st.write("**Features:** 4 input parameters")
                st.write("**Hydrocarbon Types:** 46 different fuels")
        
        # About section
        with st.expander("📘 About", expanded=False):
            st.write("This application uses Random Forest regression to predict Laminar Burning Velocity (LBV) with high accuracy.")
            st.write("**Key Features:**")
            st.write("• ±2 cm/s prediction accuracy")
            st.write("• Dynamic input validation")
            st.write("• Unit conversion (cm/s ↔ m/s)")
            st.write("• Real experimental data")
            st.write("• Cross-validated model")
        
        # Developers section
        with st.expander("👨‍💻 Developers", expanded=False):
            st.subheader("Development Team")
            st.write("**Final Year Mechanical Engineering Students**")
            st.write("Pimpri Chinchwad College of Engineering, Ravet")
            st.write("Pune, Maharashtra")
            
            st.markdown("---")
            st.subheader("Team Members")
            
            st.write("**Shekhar Sonar**")
            st.write("📧 shekharsonar641@gmail.com")
            
            st.write("**Sujal Fiske**")  
            st.write("📧 sujal.fiske_mech24@pccoer.in")
            
            st.write("**Karan Shinde**")
            st.write("📧 karan.shinde_mech23@pccoer.in")
        
        # Mentor/Project Guide section
        with st.expander("🎓 Mentor / Project Guide", expanded=False):
            st.subheader("Project Supervisor")
            st.write("**Shawnam**")
            st.write("📧 shawnam.ae111@gmail.com")
            st.write("🏢 Department of Aerospace Engineering")
            st.write("🏛️ Indian Institute of Technology Bombay")
            st.write("📍 Mumbai 400076, India")
    
    # Main content - single column layout
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("🔧 Input Parameters")
    
    # First row - Hydrocarbon selection (full width)
    hydrocarbon = st.selectbox(
        "Hydrocarbon",
        options=hydrocarbon_options,
        index=0 if hydrocarbon_options else 0,
        help="Select the type of hydrocarbon fuel"
    )
    
    # Get dynamic ranges for selected hydrocarbon
    if hydrocarbon:
        ranges = predictor.get_hydrocarbon_ranges(hydrocarbon)
        temp_min, temp_max = ranges['temperature']['min'], ranges['temperature']['max']
        eq_min, eq_max = ranges['equiv_ratio']['min'], ranges['equiv_ratio']['max']  
        press_min, press_max = ranges['pressure']['min'], ranges['pressure']['max']
    else:
        temp_min, temp_max = 300.0, 750.0
        eq_min, eq_max = 0.1, 2.4
        press_min, press_max = 1.0, 10.0
    
    # Second row - Three equal columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.number_input(
            "Initial Temperature (K)",
            min_value=temp_min,
            max_value=temp_max,
            value=min(450.0, temp_max) if temp_max >= 450.0 else temp_min,
            step=1.0,
            help=f"Temperature range for {hydrocarbon}: {temp_min:.1f}-{temp_max:.1f} K"
        )
        st.caption(f"Valid range: {temp_min:.1f} - {temp_max:.1f} K")
    
    with col2:
        equiv_ratio = st.number_input(
            "Equivalence Ratio (φ)",
            min_value=eq_min,
            max_value=eq_max,
            value=min(0.7, eq_max) if eq_max >= 0.7 else eq_min,
            step=0.01,
            help=f"Equivalence ratio range for {hydrocarbon}: {eq_min:.2f}-{eq_max:.2f}"
        )
        st.caption(f"Valid range: {eq_min:.2f} - {eq_max:.2f}")
    
    with col3:
        pressure = st.number_input(
            "Pressure (atm)",
            min_value=float(press_min),
            max_value=float(press_max),
            value=float(min(1.0, press_max) if press_max >= 1.0 else press_min),
            step=0.1,
            help=f"Pressure range for {hydrocarbon}: {press_min:.1f}-{press_max:.1f} atm"
        )
        st.caption(f"Valid range: {press_min:.1f} - {press_max:.1f} atm")
    
    # Third row - Controls
    col4, col5, col6 = st.columns([1, 1, 2])
    
    with col4:
        # Unit conversion toggle
        convert_to_ms = st.toggle("Convert to m/s", value=False, help="Toggle between cm/s and m/s units")
    
    with col6:
        # Predict button
        predict_clicked = st.button("🔥 Predict LBV", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle prediction
    if predict_clicked:
        # Validate inputs
        errors = predictor.validate_inputs(hydrocarbon, temperature, equiv_ratio, pressure)
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Make prediction
            with st.spinner("Calculating LBV..."):
                lbv_prediction = predictor.predict_lbv(hydrocarbon, temperature, equiv_ratio, pressure)
            
            if lbv_prediction is not None:
                # Convert units if needed
                if convert_to_ms:
                    lbv_display = convert_units(lbv_prediction, "cm/s", "m/s")
                    unit = "m/s"
                else:
                    lbv_display = lbv_prediction
                    unit = "cm/s"
                
                # Display result
                st.markdown(f"""
                <div class="result-container">
                    <h2>🎯 Prediction Result</h2>
                    <div class="result-value">{lbv_display:.4f} {unit}</div>
                    <p>Hydrocarbon: <strong>{hydrocarbon}</strong></p>
                    <p>T: {temperature:.1f} K | φ: {equiv_ratio:.2f} | P: {pressure:.1f} atm</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add to history
                prediction_record = {
                    'Hydrocarbon': hydrocarbon,
                    'Temperature (K)': temperature,
                    'Equivalence Ratio': equiv_ratio,
                    'Pressure (atm)': pressure,
                    f'LBV ({unit})': lbv_display,
                    'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.prediction_history.append(prediction_record)
                
                # Keep only last 50 predictions
                if len(st.session_state.prediction_history) > 50:
                    st.session_state.prediction_history = st.session_state.prediction_history[-50:]
    
    # Display prediction history
    if st.session_state.show_history and st.session_state.prediction_history:
        st.subheader("📋 Prediction History")
        
        # Convert to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display in reverse order (most recent first)
        history_df = history_df.iloc[::-1].reset_index(drop=True)
        
        st.dataframe(history_df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <p>🔬 Laminar Burning Velocity Prediction System | Built with Streamlit & Random Forest</p>
        <p>⚡ Accurate LBV predictions within ±2 cm/s | 🎯 Model R² Score: >0.90</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
