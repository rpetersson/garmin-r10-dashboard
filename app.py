import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Helper function for smart x-axis tick spacing
def get_smart_dtick(num_shots):
    """Calculate appropriate tick spacing based on number of shots"""
    if num_shots <= 10:
        return 1  # Show every shot
    elif num_shots <= 25:
        return 2  # Show every 2nd shot
    elif num_shots <= 50:
        return 5  # Show every 5th shot
    elif num_shots <= 100:
        return 10  # Show every 10th shot
    else:
        return 20  # Show every 20th shot for very large datasets

def get_smart_xaxis_config(num_shots):
    """Get optimized x-axis configuration based on number of shots"""
    dtick = get_smart_dtick(num_shots)
    
    config = {
        'dtick': dtick,
        'showgrid': True,
        'gridwidth': 1,
        'gridcolor': 'rgba(128,128,128,0.2)'
    }
    
    # For larger datasets, add some spacing and cleaner formatting
    if num_shots > 50:
        config.update({
            'tickangle': 0,  # Keep horizontal
            'tickfont': {'size': 10}
        })
    
    return config

st.set_page_config(page_title="Garmin Approach Dashboard", layout="wide")

st.title("🏌️ Garmin Approach Dashboard")
st.markdown("Upload your Garmin Approach CSV files to analyze golf swing data and visualize performance trends.")

uploaded_files = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Load and combine data
    dfs = []
    for file in uploaded_files:
        df_temp = pd.read_csv(file)
        df_temp['Source File'] = file.name
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean numeric columns
    numeric_columns = ['Club Speed', 'Ball Speed', 'Smash Factor', 'Launch Angle', 
                      'Launch Direction', 'Carry Distance', 'Total Distance', 
                      'Attack Angle', 'Club Path', 'Club Face', 'Face to Path',
                      'Backspin', 'Sidespin', 'Spin Rate', 'Spin Axis', 'Apex Height',
                      'Carry Deviation Angle', 'Carry Deviation Distance',
                      'Total Deviation Angle', 'Total Deviation Distance',
                      'Air Density', 'Temperature', 'Air Pressure', 'Relative Humidity']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to string first, then handle numeric conversion
            df[col] = df[col].astype(str)
            # Remove any units like [m], [deg], [km/h], etc.
            df[col] = df[col].str.replace(r'\[.*?\]', '', regex=True)
            # Handle concatenated values by taking the first number
            df[col] = df[col].str.split().str[0]
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date column to datetime and sort chronologically across all files
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Sort by date to get proper chronological order across all sessions
        df = df.sort_values('Date').reset_index(drop=True)
        # Calculate cumulative shot number across all sessions
        df['Shot Number'] = range(1, len(df) + 1)
        # Add session information
        df['Session'] = df.groupby(df['Date'].dt.date).ngroup() + 1
        df['Session Date'] = df['Date'].dt.date
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shots", len(df))
    with col2:
        st.metric("Files Uploaded", len(uploaded_files))
    with col3:
        if 'Club Type' in df.columns:
            st.metric("Unique Clubs", df['Club Type'].nunique())
    with col4:
        if 'Session' in df.columns:
            st.metric("Sessions", df['Session'].nunique())

    # Show session breakdown
    if 'Session' in df.columns and df['Session'].nunique() > 1:
        st.subheader("📅 Session Overview")
        session_summary = df.groupby(['Session', 'Session Date']).agg({
            'Shot Number': 'count',
            'Date': ['min', 'max']
        }).round(2)
        session_summary.columns = ['Total Shots', 'First Shot', 'Last Shot']
        st.dataframe(session_summary)

    # Show data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(10))

    # Club selection
    if 'Club Type' in df.columns:
        club_list = [club for club in df['Club Type'].unique() if pd.notna(club)]
        club_list.sort()
    else:
        club_list = []
    
    if club_list:
        selected_club = st.selectbox("Select Club Type", ["All Clubs"] + club_list)
        
        if selected_club == "All Clubs":
            club_df = df
            title_suffix = "All Clubs"
        else:
            club_df = df[df['Club Type'] == selected_club]
            title_suffix = selected_club

        # Outlier Detection and Removal
        st.subheader("🔧 Data Cleaning Options")
        
        col1, col2 = st.columns(2)
        with col1:
            remove_outliers = st.checkbox("Remove Outliers/Mishits", value=False, 
                                        help="Remove shots that are clearly mishits or measurement errors")
        
        with col2:
            outlier_method = st.selectbox("Outlier Detection Method", 
                                        ["IQR (Interquartile Range)", "Z-Score", "Golf-Specific Rules"],
                                        help="Choose how to identify outliers")
        
        if remove_outliers and len(club_df) > 0:
            original_count = len(club_df)
            
            if outlier_method == "IQR (Interquartile Range)":
                # Use IQR method for key metrics
                key_metrics = ['Carry Distance', 'Ball Speed', 'Club Speed', 'Smash Factor']
                
                for metric in key_metrics:
                    if metric in club_df.columns and club_df[metric].notna().sum() > 5:
                        Q1 = club_df[metric].quantile(0.25)
                        Q3 = club_df[metric].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        club_df = club_df[(club_df[metric] >= lower_bound) & (club_df[metric] <= upper_bound)]
            
            elif outlier_method == "Z-Score":
                # Use Z-score method (remove shots more than 2.5 standard deviations away)
                key_metrics = ['Carry Distance', 'Ball Speed', 'Club Speed', 'Smash Factor']
                
                for metric in key_metrics:
                    if metric in club_df.columns and club_df[metric].notna().sum() > 5:
                        z_scores = np.abs(stats.zscore(club_df[metric].dropna()))
                        threshold = 2.5
                        # Get indices of non-outliers
                        valid_indices = club_df[metric].dropna().index[z_scores <= threshold]
                        club_df = club_df.loc[club_df.index.isin(valid_indices)]
            
            elif outlier_method == "Golf-Specific Rules":
                # Remove clearly unrealistic golf shots
                initial_df = club_df.copy()
                
                # Smash Factor: Should be between 0.8 and 1.8 (theoretical max ~1.5)
                if 'Smash Factor' in club_df.columns:
                    club_df = club_df[(club_df['Smash Factor'].isna()) | 
                                    ((club_df['Smash Factor'] >= 0.8) & (club_df['Smash Factor'] <= 1.8))]
                
                # Ball Speed: Should be reasonable for the club type
                if 'Ball Speed' in club_df.columns:
                    if 'Driver' in title_suffix:
                        # Driver ball speeds typically 200-320 km/h
                        club_df = club_df[(club_df['Ball Speed'].isna()) | 
                                        ((club_df['Ball Speed'] >= 150) & (club_df['Ball Speed'] <= 350))]
                    elif any(iron in title_suffix for iron in ['Iron', '7', '6', '8', '9']):
                        # Iron ball speeds typically 150-250 km/h
                        club_df = club_df[(club_df['Ball Speed'].isna()) | 
                                        ((club_df['Ball Speed'] >= 100) & (club_df['Ball Speed'] <= 280))]
                    elif any(wedge in title_suffix for wedge in ['Wedge', 'SW', 'LW', 'GW', 'PW']):
                        # Wedge ball speeds typically 100-200 km/h
                        club_df = club_df[(club_df['Ball Speed'].isna()) | 
                                        ((club_df['Ball Speed'] >= 50) & (club_df['Ball Speed'] <= 220))]
                
                # Carry Distance: Remove unrealistic distances
                if 'Carry Distance' in club_df.columns:
                    if 'Driver' in title_suffix:
                        # Driver carries typically 150-350m
                        club_df = club_df[(club_df['Carry Distance'].isna()) | 
                                        ((club_df['Carry Distance'] >= 100) & (club_df['Carry Distance'] <= 400))]
                    elif any(iron in title_suffix for iron in ['7 Iron']):
                        # 7-Iron carries typically 120-200m
                        club_df = club_df[(club_df['Carry Distance'].isna()) | 
                                        ((club_df['Carry Distance'] >= 80) & (club_df['Carry Distance'] <= 250))]
                    elif any(wedge in title_suffix for wedge in ['Wedge', 'SW', 'LW', 'GW', 'PW']):
                        # Wedge carries typically 50-150m
                        club_df = club_df[(club_df['Carry Distance'].isna()) | 
                                        ((club_df['Carry Distance'] >= 20) & (club_df['Carry Distance'] <= 200))]
                
                # Launch Angle: Remove unrealistic angles
                if 'Launch Angle' in club_df.columns:
                    club_df = club_df[(club_df['Launch Angle'].isna()) | 
                                    ((club_df['Launch Angle'] >= -10) & (club_df['Launch Angle'] <= 60))]
                
                # Side Spin: Remove extreme values (> 3000 rpm is very rare)
                if 'Sidespin' in club_df.columns:
                    club_df = club_df[(club_df['Sidespin'].isna()) | 
                                    (club_df['Sidespin'].abs() <= 3000)]
            
            removed_count = original_count - len(club_df)
            if removed_count > 0:
                st.info(f"🧹 Removed {removed_count} outlier shots ({removed_count/original_count*100:.1f}% of data)")
                st.caption(f"Analyzing {len(club_df)} shots after outlier removal")
            else:
                st.success("✅ No outliers detected in the data")
        
        # Add club-specific shot numbering to eliminate gaps in trend charts
        if len(club_df) > 0:
            club_df = club_df.copy()  # Make a copy to avoid modifying original data
            club_df['Club Shot Number'] = range(1, len(club_df) + 1)
            
            # Add shot shape classification
            def classify_shot_shape(row):
                """Classify shot shape based on sidespin and ball flight"""
                sidespin = row.get('Sidespin', 0)
                carry_deviation = row.get('Carry Deviation Angle', 0)
                
                if pd.isna(sidespin) or pd.isna(carry_deviation):
                    return 'Unknown'
                
                # Convert to numbers if they're strings
                try:
                    sidespin = float(sidespin)
                    carry_deviation = float(carry_deviation)
                except (ValueError, TypeError):
                    return 'Unknown'
                
                # Classify based on sidespin and final ball position
                # Positive sidespin = right spin, negative = left spin
                # Positive deviation = right of target, negative = left of target
                
                if abs(sidespin) < 200 and abs(carry_deviation) < 2:
                    return 'Straight'
                elif sidespin > 0:  # Right spin
                    if carry_deviation > 5:  # Significant right curve
                        return 'Slice' if sidespin > 800 else 'Fade'
                    elif carry_deviation < -2:  # Ball curves back left despite right spin
                        return 'Pull'
                    else:
                        return 'Fade'
                elif sidespin < 0:  # Left spin
                    if carry_deviation < -5:  # Significant left curve
                        return 'Hook' if sidespin < -800 else 'Draw'
                    elif carry_deviation > 2:  # Ball curves back right despite left spin
                        return 'Push'
                    else:
                        return 'Draw'
                else:
                    return 'Straight'
            
            # Apply shot shape classification
            club_df['Shot Shape'] = club_df.apply(classify_shot_shape, axis=1)

        st.subheader(f"📈 Performance Analysis - {title_suffix}")
        
        if len(club_df) == 0:
            st.warning("No data available for the selected club.")
        else:
            # Key metrics
            metrics_cols = st.columns(4)
            
            if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
                avg_distance = club_df['Carry Distance'].mean()
                
                # Calculate trend for carry distance
                if len(club_df) >= 10:
                    # Split data into first and last thirds for trend comparison
                    third_size = len(club_df) // 3
                    first_third = club_df.head(third_size)['Carry Distance'].mean()
                    last_third = club_df.tail(third_size)['Carry Distance'].mean()
                    distance_trend = last_third - first_third
                    
                    if distance_trend > 2:  # Improving by more than 2m
                        trend_arrow = "📈"
                        trend_color = "green"
                        trend_text = f"+{distance_trend:.1f}m"
                    elif distance_trend < -2:  # Declining by more than 2m
                        trend_arrow = "📉"
                        trend_color = "red"
                        trend_text = f"{distance_trend:.1f}m"
                    else:  # Maintaining
                        trend_arrow = "➡️"
                        trend_color = "blue"
                        trend_text = "Stable"
                else:
                    trend_arrow = ""
                    trend_color = "black"
                    trend_text = ""
                
                with metrics_cols[0]:
                    st.metric("Avg Carry Distance", f"{avg_distance:.1f}m")
                    if trend_text:
                        st.markdown(f"<span style='color:{trend_color}'>{trend_arrow} {trend_text}</span>", unsafe_allow_html=True)
            
            if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                avg_smash = club_df['Smash Factor'].mean()
                optimal_smash = 1.30
                delta_smash = avg_smash - optimal_smash
                delta_text = f"({delta_smash:+.3f})" if abs(delta_smash) > 0.001 else ""
                with metrics_cols[1]:
                    st.metric("Avg Smash Factor", f"{avg_smash:.3f}", delta=delta_text, help="Optimal smash factor is around 1.30")
            
            if 'Club Speed' in club_df.columns and club_df['Club Speed'].notna().any():
                avg_club_speed = club_df['Club Speed'].mean()
                with metrics_cols[2]:
                    st.metric("Avg Club Speed", f"{avg_club_speed:.1f} km/h")
            
            if 'Total Distance' in club_df.columns and club_df['Total Distance'].notna().any():
                avg_total_distance = club_df['Total Distance'].mean()
                
                # Calculate trend for total distance
                if len(club_df) >= 10:
                    # Split data into first and last thirds for trend comparison
                    third_size = len(club_df) // 3
                    first_third = club_df.head(third_size)['Total Distance'].mean()
                    last_third = club_df.tail(third_size)['Total Distance'].mean()
                    total_trend = last_third - first_third
                    
                    if total_trend > 2:  # Improving by more than 2m
                        total_arrow = "📈"
                        total_color = "green"
                        total_text = f"+{total_trend:.1f}m"
                    elif total_trend < -2:  # Declining by more than 2m
                        total_arrow = "📉"
                        total_color = "red"
                        total_text = f"{total_trend:.1f}m"
                    else:  # Maintaining
                        total_arrow = "➡️"
                        total_color = "blue"
                        total_text = "Stable"
                else:
                    total_arrow = ""
                    total_color = "black"
                    total_text = ""
                
                with metrics_cols[3]:
                    st.metric("Avg Total Distance", f"{avg_total_distance:.1f}m")
                    if total_text:
                        st.markdown(f"<span style='color:{total_color}'>{total_arrow} {total_text}</span>", unsafe_allow_html=True)
            elif 'Launch Angle' in club_df.columns and club_df['Launch Angle'].notna().any():
                avg_launch = club_df['Launch Angle'].mean()
                with metrics_cols[3]:
                    st.metric("Avg Launch Angle", f"{avg_launch:.1f}°")

            # Performance trends
            trend_stats = [
                ('Club Speed', 'Club Speed', 'km/h'),
                ('Ball Speed', 'Ball Speed', 'km/h'),
                ('Smash Factor', 'Smash Factor', ''),
                ('Launch Angle', 'Launch Angle', '°'),
                ('Attack Angle', 'Attack Angle', '°'),
                ('Carry Distance', 'Carry Distance', 'm'),
                ('Total Distance', 'Total Distance', 'm'),
            ]
            
            # Create subplot layout
            available_stats = [stat for stat in trend_stats if stat[1] in club_df.columns]
            
            # Add smash factor efficiency analysis
            if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                st.subheader("⚡ Smash Factor Efficiency")
                
                optimal_smash = 1.30
                avg_smash = club_df['Smash Factor'].mean()
                efficiency = (avg_smash / optimal_smash) * 100
                
                efficiency_cols = st.columns(4)
                with efficiency_cols[0]:
                    st.metric("Average Smash Factor", f"{avg_smash:.3f}")
                with efficiency_cols[1]:
                    st.metric("Optimal Target", f"{optimal_smash:.3f}")
                with efficiency_cols[2]:
                    delta = avg_smash - optimal_smash
                    delta_color = "green" if abs(delta) < 0.05 else ("orange" if abs(delta) < 0.1 else "red")
                    st.metric("Difference from Optimal", f"{delta:+.3f}")
                with efficiency_cols[3]:
                    efficiency_color = "green" if efficiency >= 95 else ("orange" if efficiency >= 85 else "red")
                    st.metric("Efficiency", f"{efficiency:.1f}%")
                
                # Smash factor consistency analysis
                smash_std = club_df['Smash Factor'].std()
                consistency_text = "Excellent" if smash_std < 0.05 else ("Good" if smash_std < 0.1 else "Needs Improvement")
                consistency_color = "green" if smash_std < 0.05 else ("orange" if smash_std < 0.1 else "red")
                
                st.markdown(f"**Consistency:** <span style='color:{consistency_color}'>{consistency_text}</span> (σ = {smash_std:.3f})", unsafe_allow_html=True)
                
                # Show shots within optimal range
                optimal_range = club_df[(club_df['Smash Factor'] >= 1.25) & (club_df['Smash Factor'] <= 1.35)]
                percentage_optimal = (len(optimal_range) / len(club_df)) * 100
                st.info(f"🎯 {percentage_optimal:.1f}% of shots within optimal range (1.25-1.35)")
            
            if available_stats:
                # Create tabs for different views
                tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Summary", "📈 Performance Trends", "📊 Accuracy Analysis", "🎯 Shot Pattern", "🌪️ Spin Analysis", "🌤️ Environmental Impact"])
                
                with tab0:
                    st.subheader("📊 Performance Summary")
                    
                    # Data summary information
                    data_info_cols = st.columns(4)
                    
                    with data_info_cols[0]:
                        total_shots = len(club_df)
                        st.metric("Total Shots", f"{total_shots}")
                    
                    with data_info_cols[1]:
                        if 'Session' in club_df.columns:
                            total_sessions = club_df['Session'].nunique()
                            st.metric("Sessions", f"{total_sessions}")
                        else:
                            st.metric("Sessions", "1")
                    
                    with data_info_cols[2]:
                        if 'Date' in club_df.columns and club_df['Date'].notna().any():
                            # Calculate date range
                            dates = pd.to_datetime(club_df['Date'], errors='coerce')
                            if dates.notna().any():
                                date_range = (dates.max() - dates.min()).days + 1
                                if date_range == 1:
                                    st.metric("Time Frame", "1 day")
                                else:
                                    st.metric("Time Frame", f"{date_range} days")
                            else:
                                st.metric("Time Frame", "N/A")
                        else:
                            st.metric("Time Frame", "N/A")
                    
                    with data_info_cols[3]:
                        if 'Date' in club_df.columns and club_df['Date'].notna().any():
                            dates = pd.to_datetime(club_df['Date'], errors='coerce')
                            if dates.notna().any():
                                latest_date = dates.max().strftime('%Y-%m-%d')
                                st.metric("Latest Session", latest_date)
                            else:
                                st.metric("Latest Session", "N/A")
                        else:
                            st.metric("Latest Session", "N/A")
                    
                    st.markdown("---")  # Add separator line
                    
                    # Overall performance metrics in a grid
                    summary_cols = st.columns(3)
                    
                    # Distance Performance
                    with summary_cols[0]:
                        st.markdown("### 🎯 **Distance Performance**")
                        if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
                            avg_carry = club_df['Carry Distance'].mean()
                            max_carry = club_df['Carry Distance'].max()
                            std_carry = club_df['Carry Distance'].std()
                            
                            st.metric("Average Carry", f"{avg_carry:.1f}m")
                            st.metric("Best Shot", f"{max_carry:.1f}m")
                            st.metric("Consistency (±)", f"{std_carry:.1f}m")
                            
                            # Distance rating based on club type
                            if 'Driver' in title_suffix:
                                if avg_carry > 250: rating = "Excellent 🔥"
                                elif avg_carry > 220: rating = "Good 👍"
                                elif avg_carry > 180: rating = "Average 📊"
                                else: rating = "Needs Work 💪"
                            elif any(iron in title_suffix for iron in ['7 Iron']):
                                if avg_carry > 160: rating = "Excellent 🔥"
                                elif avg_carry > 140: rating = "Good 👍"
                                elif avg_carry > 120: rating = "Average 📊"
                                else: rating = "Needs Work 💪"
                            else:
                                rating = "Data Available 📊"
                            
                            st.caption(f"Rating: {rating}")
                    
                    # Efficiency Performance  
                    with summary_cols[1]:
                        st.markdown("### ⚡ **Efficiency**")
                        if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                            avg_smash = club_df['Smash Factor'].mean()
                            optimal_shots = club_df[(club_df['Smash Factor'] >= 1.25) & (club_df['Smash Factor'] <= 1.35)]
                            efficiency_pct = (len(optimal_shots) / len(club_df)) * 100
                            
                            st.metric("Avg Smash Factor", f"{avg_smash:.3f}")
                            st.metric("Optimal Range %", f"{efficiency_pct:.1f}%")
                            
                            if avg_smash >= 1.28: rating = "Excellent 🔥"
                            elif avg_smash >= 1.22: rating = "Good 👍"
                            elif avg_smash >= 1.15: rating = "Average 📊"
                            else: rating = "Needs Work 💪"
                            
                            st.caption(f"Rating: {rating}")
                        
                        if 'Ball Speed' in club_df.columns and club_df['Ball Speed'].notna().any():
                            avg_ball_speed = club_df['Ball Speed'].mean()
                            st.metric("Avg Ball Speed", f"{avg_ball_speed:.1f} km/h")
                    
                    # Accuracy Performance
                    with summary_cols[2]:
                        st.markdown("### 🎯 **Accuracy**")
                        if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                            direction_std = club_df['Launch Direction'].std()
                            straight_shots = club_df[club_df['Launch Direction'].abs() <= 5]
                            accuracy_pct = (len(straight_shots) / len(club_df)) * 100
                            
                            st.metric("Direction Spread", f"±{direction_std:.1f}°")
                            st.metric("Straight Shots %", f"{accuracy_pct:.1f}%")
                            
                            if direction_std <= 8: rating = "Excellent 🔥"
                            elif direction_std <= 12: rating = "Good 👍"
                            elif direction_std <= 18: rating = "Average 📊"
                            else: rating = "Needs Work 💪"
                            
                            st.caption(f"Rating: {rating}")
                        
                        # Iron-specific attack angle metric (only show for irons)
                        if ('Attack Angle' in club_df.columns and club_df['Attack Angle'].notna().any() and
                            any(iron in title_suffix for iron in ['Iron', '7 Iron', '6 Iron', '8 Iron', '9 Iron', '5 Iron', '4 Iron'])):
                            avg_attack_angle = club_df['Attack Angle'].mean()
                            optimal_iron_shots = club_df[(club_df['Attack Angle'] >= -4) & (club_df['Attack Angle'] <= -2)]
                            iron_angle_pct = (len(optimal_iron_shots) / len(club_df)) * 100
                            
                            st.metric("Avg Attack Angle", f"{avg_attack_angle:.1f}°")
                            st.metric("Optimal Range %", f"{iron_angle_pct:.1f}%")
                            
                            # Rating based on how close to -3°
                            distance_from_ideal = abs(avg_attack_angle - (-3))
                            if distance_from_ideal <= 0.5: rating = "Perfect 🔥"
                            elif distance_from_ideal <= 1.0: rating = "Excellent 👍"
                            elif distance_from_ideal <= 2.0: rating = "Good 📊"
                            else: rating = "Needs Work 💪"
                            
                            st.caption(f"Iron Strike: {rating}")
                            st.caption("Target: -3° (hitting down)")
                        
                        if 'Sidespin' in club_df.columns and club_df['Sidespin'].notna().any():
                            avg_sidespin = club_df['Sidespin'].abs().mean()
                            st.metric("Avg Sidespin", f"{avg_sidespin:.0f} rpm")
                    
                    # Shot Shape Analysis
                    if 'Shot Shape' in club_df.columns:
                        st.subheader("🎯 Shot Shape Analysis")
                        
                        # Count shot shapes
                        shape_counts = club_df['Shot Shape'].value_counts()
                        
                        # Create shot shape distribution chart
                        if len(shape_counts) > 0:
                            shape_cols = st.columns(2)
                            
                            with shape_cols[0]:
                                # Shot shape pie chart
                                fig = px.pie(
                                    values=shape_counts.values,
                                    names=shape_counts.index,
                                    title="Shot Shape Distribution",
                                    color_discrete_map={
                                        'Straight': '#28a745',
                                        'Draw': '#17a2b8', 
                                        'Hook': '#dc3545',
                                        'Fade': '#6c757d',
                                        'Slice': '#fd7e14',
                                        'Push': '#ffc107',
                                        'Pull': '#e83e8c'
                                    }
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with shape_cols[1]:
                                # Shot shape statistics
                                total_shots = len(club_df)
                                st.write("**Shot Shape Breakdown:**")
                                
                                for shape, count in shape_counts.items():
                                    percentage = (count / total_shots) * 100
                                    if shape == 'Straight':
                                        emoji = "🎯"
                                        color = "green"
                                    elif shape in ['Draw', 'Fade']:
                                        emoji = "✅"
                                        color = "blue"
                                    elif shape in ['Hook', 'Slice']:
                                        emoji = "⚠️"
                                        color = "orange"
                                    else:
                                        emoji = "📊"
                                        color = "gray"
                                    
                                    st.markdown(f"{emoji} **{shape}**: {count} shots ({percentage:.1f}%)")
                                
                                # Shot shape consistency rating
                                good_shapes = shape_counts.get('Straight', 0) + shape_counts.get('Draw', 0) + shape_counts.get('Fade', 0)
                                consistency_percentage = (good_shapes / total_shots) * 100
                                
                                if consistency_percentage >= 70:
                                    st.success(f"🎯 **Excellent shot shape control**: {consistency_percentage:.1f}% controlled shots")
                                elif consistency_percentage >= 50:
                                    st.info(f"👍 **Good shot shape control**: {consistency_percentage:.1f}% controlled shots")
                                else:
                                    st.warning(f"⚠️ **Work on shot shape**: {consistency_percentage:.1f}% controlled shots")
                    
                    # Detailed Recommendations
                    st.subheader("🎯 Personalized Recommendations")
                    
                    recommendations = []
                    priority_areas = []
                    
                    # Smash Factor Analysis
                    if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                        avg_smash = club_df['Smash Factor'].mean()
                        if avg_smash < 1.20:
                            priority_areas.append("Smash Factor")
                            recommendations.append({
                                "icon": "⚡",
                                "title": "Improve Smash Factor (Priority)",
                                "issue": f"Your average smash factor is {avg_smash:.3f}, below optimal range",
                                "solution": "Focus on center face contact. Practice with impact tape or use alignment sticks for consistent setup.",
                                "target": "Target: 1.25-1.35 range"
                            })
                        elif avg_smash < 1.25:
                            recommendations.append({
                                "icon": "⚡",
                                "title": "Optimize Smash Factor",
                                "issue": f"Your smash factor of {avg_smash:.3f} has room for improvement",
                                "solution": "Work on consistent ball striking. Check your grip pressure and swing tempo.",
                                "target": "Target: 1.30+ for maximum efficiency"
                            })
                    
                    # Distance Consistency
                    if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
                        distance_std = club_df['Carry Distance'].std()
                        if distance_std > 15:
                            priority_areas.append("Distance Control")
                            recommendations.append({
                                "icon": "📏",
                                "title": "Improve Distance Control",
                                "issue": f"Distance spread of ±{distance_std:.1f}m indicates inconsistent contact",
                                "solution": "Practice tempo drills and work on consistent swing length. Consider lesson on swing plane.",
                                "target": "Target: ±10m or less for good consistency"
                            })
                    
                    # Direction Accuracy
                    if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                        direction_std = club_df['Launch Direction'].std()
                        if direction_std > 15:
                            priority_areas.append("Direction Control")
                            recommendations.append({
                                "icon": "🎯",
                                "title": "Improve Direction Control",
                                "issue": f"Direction spread of ±{direction_std:.1f}° suggests alignment or face control issues",
                                "solution": "Check setup alignment and practice square club face at impact. Use alignment sticks during practice.",
                                "target": "Target: ±10° for good accuracy"
                            })
                    
                    # Launch Conditions
                    if 'Launch Angle' in club_df.columns and club_df['Launch Angle'].notna().any():
                        avg_launch = club_df['Launch Angle'].mean()
                        if 'Driver' in title_suffix:
                            if avg_launch < 10 or avg_launch > 20:
                                recommendations.append({
                                    "icon": "📐",
                                    "title": "Optimize Launch Angle",
                                    "issue": f"Driver launch angle of {avg_launch:.1f}° is outside optimal range",
                                    "solution": "Adjust tee height and ball position. Consider loft adjustment or different driver.",
                                    "target": "Target: 12-17° for optimal carry"
                                })
                        elif any(iron in title_suffix for iron in ['Iron']):
                            if avg_launch < 12 or avg_launch > 30:
                                recommendations.append({
                                    "icon": "📐",
                                    "title": "Optimize Launch Angle",
                                    "issue": f"Iron launch angle of {avg_launch:.1f}° could be optimized",
                                    "solution": "Check ball position and angle of attack. Ball should be slightly forward of center for irons.",
                                    "target": "Target: 15-25° for optimal trajectory"
                                })
                    
                    # Spin Analysis
                    if 'Backspin' in club_df.columns and club_df['Backspin'].notna().any():
                        avg_backspin = club_df['Backspin'].mean()
                        if 'Driver' in title_suffix:
                            if avg_backspin > 3000:
                                recommendations.append({
                                    "icon": "🌪️",
                                    "title": "Reduce Driver Backspin",
                                    "issue": f"Backspin of {avg_backspin:.0f} rpm is too high for driver",
                                    "solution": "Hit up on the ball more (positive attack angle). Tee ball higher and move it forward in stance.",
                                    "target": "Target: 2000-2500 rpm for maximum distance"
                                })
                        elif any(iron in title_suffix for iron in ['Iron']):
                            if avg_backspin < 4000 or avg_backspin > 8000:
                                recommendations.append({
                                    "icon": "🌪️",
                                    "title": "Optimize Iron Backspin",
                                    "issue": f"Iron backspin of {avg_backspin:.0f} rpm is outside optimal range",
                                    "solution": "Work on clean contact and proper descending blow with irons.",
                                    "target": "Target: 5000-7000 rpm for good stopping power"
                                })
                    
                    # Shot Shape Recommendations
                    if 'Shot Shape' in club_df.columns and len(club_df) > 0:
                        shape_counts = club_df['Shot Shape'].value_counts()
                        total_shots = len(club_df)
                        
                        # Check for problematic shot shapes
                        slice_percentage = (shape_counts.get('Slice', 0) / total_shots) * 100
                        hook_percentage = (shape_counts.get('Hook', 0) / total_shots) * 100
                        push_pull_percentage = ((shape_counts.get('Push', 0) + shape_counts.get('Pull', 0)) / total_shots) * 100
                        
                        if slice_percentage > 30:
                            priority_areas.append("Slice Control")
                            recommendations.append({
                                "icon": "🌪️",
                                "title": "Fix Slice Pattern (Priority)",
                                "issue": f"{slice_percentage:.1f}% of your shots are slices",
                                "solution": "Strengthen your grip, work on swing path (more inside-out), and close club face at impact. Check your setup alignment.",
                                "target": "Target: <15% slices for better accuracy"
                            })
                        elif slice_percentage > 15:
                            recommendations.append({
                                "icon": "➡️",
                                "title": "Reduce Slice Tendency",
                                "issue": f"{slice_percentage:.1f}% of your shots are slices",
                                "solution": "Focus on club face control and slightly more inside swing path.",
                                "target": "Target: <10% slices for consistent ball flight"
                            })
                        
                        if hook_percentage > 30:
                            priority_areas.append("Hook Control")
                            recommendations.append({
                                "icon": "↩️",
                                "title": "Control Hook Pattern (Priority)",
                                "issue": f"{hook_percentage:.1f}% of your shots are hooks",
                                "solution": "Weaken your grip slightly, work on swing path (less inside-out), and square club face at impact.",
                                "target": "Target: <15% hooks for better control"
                            })
                        elif hook_percentage > 15:
                            recommendations.append({
                                "icon": "⬅️",
                                "title": "Reduce Hook Tendency",
                                "issue": f"{hook_percentage:.1f}% of your shots are hooks",
                                "solution": "Focus on club face control and slightly less inside swing path.",
                                "target": "Target: <10% hooks for consistent ball flight"
                            })
                        
                        if push_pull_percentage > 20:
                            recommendations.append({
                                "icon": "🎯",
                                "title": "Improve Swing Path Consistency",
                                "issue": f"{push_pull_percentage:.1f}% of shots are pushes/pulls",
                                "solution": "Work on swing path consistency. Focus on proper body rotation and club face alignment at impact.",
                                "target": "Target: <10% pushes/pulls for better accuracy"
                            })
                    
                    # Iron Attack Angle Recommendations (hitting down on the ball)
                    if ('Attack Angle' in club_df.columns and club_df['Attack Angle'].notna().any() and 
                        any(iron in title_suffix for iron in ['Iron', '7 Iron', '6 Iron', '8 Iron', '9 Iron', '5 Iron', '4 Iron'])):
                        avg_attack_angle = club_df['Attack Angle'].mean()
                        
                        if avg_attack_angle > -1:  # Not hitting down enough
                            priority_areas.append("Iron Attack Angle")
                            recommendations.append({
                                "icon": "⬇️",
                                "title": "Improve Iron Attack Angle (Priority)",
                                "issue": f"Your average attack angle is {avg_attack_angle:.1f}°, but irons need a downward strike",
                                "solution": "Focus on hitting down on the ball with irons. Ball position slightly back of center, hands ahead at impact, weight favoring front foot.",
                                "target": "Target: -3° attack angle for optimal iron performance"
                            })
                        elif avg_attack_angle > -2:  # Close but could be better
                            recommendations.append({
                                "icon": "📐",
                                "title": "Fine-tune Iron Attack Angle",
                                "issue": f"Attack angle of {avg_attack_angle:.1f}° is close but could be more optimal",
                                "solution": "Slightly steeper angle of attack. Focus on ball-first contact and taking a divot after the ball.",
                                "target": "Target: -3° for ideal iron compression and trajectory"
                            })
                        elif avg_attack_angle < -5:  # Too steep
                            recommendations.append({
                                "icon": "📏",
                                "title": "Reduce Excessive Downward Strike",
                                "issue": f"Attack angle of {avg_attack_angle:.1f}° is too steep for irons",
                                "solution": "Shallow out your swing slightly. Focus on sweeping through impact rather than chopping down.",
                                "target": "Target: -3° for optimal balance of compression and distance"
                            })
                        elif -4 <= avg_attack_angle <= -2:  # In the sweet spot
                            # Don't add a recommendation, but could add a positive note
                            pass
                    
                    # Display recommendations
                    if recommendations:
                        if priority_areas:
                            st.error(f"🚨 **Priority Areas**: {', '.join(priority_areas)}")
                        
                        for i, rec in enumerate(recommendations):
                            with st.expander(f"{rec['icon']} {rec['title']}", expanded=i<2):
                                st.markdown(f"**Issue:** {rec['issue']}")
                                st.markdown(f"**Solution:** {rec['solution']}")
                                st.markdown(f"**{rec['target']}**")
                    else:
                        st.success("🎉 **Excellent Performance!** Your metrics are all within optimal ranges. Keep up the great work!")
                        st.info("💡 **Maintenance Tips:** Continue regular practice to maintain consistency. Focus on small refinements rather than major changes.")
                
                with tab1:
                    # Add option to show session boundaries
                    if 'Session' in club_df.columns and club_df['Session'].nunique() > 1:
                        show_sessions = st.checkbox("Show session boundaries", value=True)
                    else:
                        show_sessions = False
                    
                    for i, (label, col, unit) in enumerate(available_stats):
                        if col in club_df.columns and club_df[col].notna().any():
                            try:
                                fig = go.Figure()
                                
                                # Add trend line
                                fig.add_trace(go.Scatter(
                                    x=club_df['Club Shot Number'],
                                    y=club_df[col],
                                    mode='lines+markers',
                                    name=label,
                                    line=dict(width=2),
                                    marker=dict(size=6),
                                    hovertemplate='<b>Shot %{x}</b><br>' +
                                                f'{label}: %{{y}}<br>' +
                                                '<extra></extra>'
                                ))
                                
                                # Add session boundaries if requested
                                if show_sessions and 'Session' in club_df.columns:
                                    # Find session changes within the filtered club data
                                    session_changes = club_df[club_df['Session'] != club_df['Session'].shift(1)]
                                    for idx, row in session_changes[1:].iterrows():  # Skip first session
                                        club_shot_num = club_df.loc[idx, 'Club Shot Number']
                                        session_name = row['Session']
                                        fig.add_vline(
                                            x=club_shot_num,
                                            line_dash="dash",
                                            line_color="gray",
                                            annotation_text=f"Session {session_name}",
                                            annotation_position="top"
                                        )
                                
                                # Add moving average if enough data points
                                if len(club_df) > 5:
                                    window = min(5, len(club_df) // 3)
                                    moving_avg = club_df[col].rolling(window=window, center=True).mean()
                                    fig.add_trace(go.Scatter(
                                        x=club_df['Club Shot Number'],
                                        y=moving_avg,
                                        mode='lines',
                                        name=f'Moving Average ({window} shots)',
                                        line=dict(width=3, dash='dash'),
                                        opacity=0.7
                                    ))
                                
                                # Add optimal zones for golf-specific metrics
                                if col == 'Smash Factor':
                                    fig.add_hline(y=1.30, line_dash="dot", line_color="green", 
                                                annotation_text="Optimal (1.30)", annotation_position="bottom right")
                                    fig.add_hrect(y0=1.25, y1=1.35, fillcolor="green", opacity=0.1, 
                                                annotation_text="Optimal Range", annotation_position="top left")
                                elif col == 'Launch Angle':
                                    # Optimal launch angle varies by club - add general guidance
                                    if 'Driver' in title_suffix:
                                        fig.add_hrect(y0=12, y1=17, fillcolor="green", opacity=0.1, 
                                                    annotation_text="Driver Optimal (12-17°)", annotation_position="top left")
                                    elif any(iron in title_suffix for iron in ['7 Iron', '6 Iron', '8 Iron', '9 Iron']):
                                        fig.add_hrect(y0=15, y1=25, fillcolor="green", opacity=0.1, 
                                                    annotation_text="Iron Optimal (15-25°)", annotation_position="top left")
                                    elif any(wedge in title_suffix for wedge in ['Wedge', 'SW', 'LW', 'GW', 'PW']):
                                        fig.add_hrect(y0=25, y1=45, fillcolor="green", opacity=0.1, 
                                                    annotation_text="Wedge Optimal (25-45°)", annotation_position="top left")
                                elif col == 'Attack Angle':
                                    # Attack angle optimal zones
                                    if 'Driver' in title_suffix:
                                        fig.add_hrect(y0=2, y1=5, fillcolor="green", opacity=0.1, 
                                                    annotation_text="Driver Optimal (+2 to +5°)", annotation_position="top left")
                                    elif any(iron in title_suffix for iron in ['Iron']):
                                        # Iron-specific attack angle guidance - hitting down is desired
                                        fig.add_hrect(y0=-5, y1=-1, fillcolor="lightgreen", opacity=0.15, 
                                                    annotation_text="Iron Optimal (-5 to -1°)", annotation_position="top left")
                                        # Highlight the ideal -3° line
                                        fig.add_hline(y=-3, line_dash="solid", line_color="darkgreen", line_width=2,
                                                    annotation_text="Ideal Iron Attack Angle (-3°)", annotation_position="bottom right")
                                        fig.add_hrect(y0=-4, y1=-2, fillcolor="green", opacity=0.2, 
                                                    annotation_text="Sweet Spot (-4 to -2°)", annotation_position="bottom left")
                                elif col == 'Club Speed':
                                    # Add reference lines for typical speeds by club
                                    if 'Driver' in title_suffix:
                                        fig.add_hline(y=150, line_dash="dot", line_color="blue", opacity=0.5,
                                                    annotation_text="Average Driver Speed", annotation_position="bottom right")
                                    elif any(iron in title_suffix for iron in ['7 Iron']):
                                        fig.add_hline(y=120, line_dash="dot", line_color="blue", opacity=0.5,
                                                    annotation_text="Average 7-Iron Speed", annotation_position="bottom right")
                                
                                # Add trend analysis for distance metrics
                                if col in ['Carry Distance', 'Total Distance'] and len(club_df) >= 10:
                                    # Calculate trend line
                                    x_vals = club_df['Club Shot Number']
                                    y_vals = club_df[col].dropna()
                                    x_vals_clean = x_vals[club_df[col].notna()]
                                    
                                    if len(y_vals) >= 5:
                                        z = np.polyfit(x_vals_clean, y_vals, 1)
                                        trend_line = np.poly1d(z)
                                        
                                        # Add subtle trend line
                                        fig.add_trace(go.Scatter(
                                            x=x_vals_clean,
                                            y=trend_line(x_vals_clean),
                                            mode='lines',
                                            name='Overall Trend',
                                            line=dict(dash='dot', color='red', width=1),
                                            opacity=0.6
                                        ))
                                        
                                        # Calculate trend direction
                                        slope = z[0]
                                        if slope > 1:  # Improving by more than 1m per shot
                                            trend_emoji = "📈"
                                            trend_color = "green"
                                        elif slope < -1:  # Declining by more than 1m per shot
                                            trend_emoji = "📉"
                                            trend_color = "red"
                                        else:
                                            trend_emoji = "➡️"
                                            trend_color = "blue"
                                        
                                        # Add trend annotation
                                        fig.add_annotation(
                                            x=x_vals_clean.iloc[-1],
                                            y=trend_line(x_vals_clean.iloc[-1]),
                                            text=f"{trend_emoji} {slope:.1f}m/shot",
                                            showarrow=True,
                                            arrowhead=2,
                                            arrowsize=1,
                                            arrowwidth=2,
                                            arrowcolor=trend_color,
                                            font=dict(color=trend_color, size=12)
                                        )
                                
                                fig.update_layout(
                                    title=f"{label} Trend Over Time",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title=f"{label} ({unit})" if unit else label,
                                    height=400,
                                    showlegend=True,
                                    xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                                )
                                
                                # Set appropriate y-axis range for Smash Factor
                                if col == 'Smash Factor':
                                    # Get data range and set reasonable bounds around optimal 1.30
                                    min_val = club_df[col].min()
                                    max_val = club_df[col].max()
                                    
                                    # Set bounds with some padding around the data, but focused on golf-relevant range
                                    y_min = max(0.8, min_val - 0.1)  # Don't go below 0.8 (unrealistic)
                                    y_max = min(1.8, max_val + 0.1)  # Don't go above 1.8 (unrealistic)
                                    
                                    fig.update_yaxes(range=[y_min, y_max])
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating chart for {label}: {str(e)}")
                                st.write(f"Data type for {col}: {club_df[col].dtype}")
                                st.write(f"Sample values: {club_df[col].head()}")
                
                with tab2:
                    st.subheader("🎯 Accuracy & Consistency Analysis")
                    
                    # Accuracy metrics
                    accuracy_cols = st.columns(3)
                    
                    # Distance accuracy
                    if 'Carry Deviation Distance' in club_df.columns and club_df['Carry Deviation Distance'].notna().any():
                        with accuracy_cols[0]:
                            avg_deviation = club_df['Carry Deviation Distance'].abs().mean()
                            st.metric("Avg Distance Deviation", f"{avg_deviation:.1f}m")
                    
                    # Direction accuracy
                    if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                        with accuracy_cols[1]:
                            direction_std = club_df['Launch Direction'].std()
                            st.metric("Direction Consistency", f"±{direction_std:.1f}°")
                    
                    # Club path consistency
                    if 'Club Path' in club_df.columns and club_df['Club Path'].notna().any():
                        with accuracy_cols[2]:
                            path_std = club_df['Club Path'].std()
                            st.metric("Club Path Consistency", f"±{path_std:.1f}°")
                    
                    # Accuracy trends over time
                    st.subheader("📈 Accuracy Improvement Trends")
                    
                    # Calculate rolling accuracy metrics
                    if len(club_df) > 10:
                        window_size = max(5, len(club_df) // 10)
                        
                        accuracy_trend_cols = st.columns(2)
                        
                        # Distance accuracy trend
                        if 'Carry Deviation Distance' in club_df.columns and club_df['Carry Deviation Distance'].notna().any():
                            with accuracy_trend_cols[0]:
                                club_df_copy = club_df.copy()
                                club_df_copy['Distance_Accuracy_Trend'] = club_df_copy['Carry Deviation Distance'].abs().rolling(
                                    window=window_size, center=True
                                ).mean()
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=club_df_copy['Club Shot Number'],
                                    y=club_df_copy['Distance_Accuracy_Trend'],
                                    mode='lines+markers',
                                    name='Distance Accuracy',
                                    line=dict(width=3, color='red'),
                                    marker=dict(size=4)
                                ))
                                
                                # Add trend line
                                if len(club_df_copy.dropna(subset=['Distance_Accuracy_Trend'])) > 5:
                                    x_vals = club_df_copy.dropna(subset=['Distance_Accuracy_Trend'])['Club Shot Number']
                                    y_vals = club_df_copy.dropna(subset=['Distance_Accuracy_Trend'])['Distance_Accuracy_Trend']
                                    z = np.polyfit(x_vals, y_vals, 1)
                                    trend_line = np.poly1d(z)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=trend_line(x_vals),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(dash='dash', color='darkred', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Calculate improvement
                                    slope = z[0]
                                    if slope < 0:
                                        trend_text = f"📈 Improving: {abs(slope):.2f}m better per shot"
                                        trend_color = "green"
                                    else:
                                        trend_text = f"📉 Declining: {slope:.2f}m worse per shot"
                                        trend_color = "red"
                                    
                                    st.markdown(f"<span style='color:{trend_color}'>{trend_text}</span>", unsafe_allow_html=True)
                                
                                fig.update_layout(
                                    title=f"Distance Accuracy Trend (Rolling {window_size}-shot average)",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title="Average Distance Deviation (m)",
                                    height=350,
                                    xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Direction consistency trend
                        if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                            with accuracy_trend_cols[1]:
                                club_df_copy = club_df.copy()
                                club_df_copy['Direction_Consistency_Trend'] = club_df_copy['Launch Direction'].rolling(
                                    window=window_size, center=True
                                ).std()
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=club_df_copy['Club Shot Number'],
                                    y=club_df_copy['Direction_Consistency_Trend'],
                                    mode='lines+markers',
                                    name='Direction Consistency',
                                    line=dict(width=3, color='blue'),
                                    marker=dict(size=4)
                                ))
                                
                                # Add trend line
                                if len(club_df_copy.dropna(subset=['Direction_Consistency_Trend'])) > 5:
                                    x_vals = club_df_copy.dropna(subset=['Direction_Consistency_Trend'])['Club Shot Number']
                                    y_vals = club_df_copy.dropna(subset=['Direction_Consistency_Trend'])['Direction_Consistency_Trend']
                                    z = np.polyfit(x_vals, y_vals, 1)
                                    trend_line = np.poly1d(z)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=trend_line(x_vals),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(dash='dash', color='darkblue', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Calculate improvement (lower is better for consistency)
                                    slope = z[0]
                                    if slope < 0:
                                        trend_text = f"📈 Improving: {abs(slope):.3f}° more consistent per shot"
                                        trend_color = "green"
                                    else:
                                        trend_text = f"📉 Declining: {slope:.3f}° less consistent per shot"
                                        trend_color = "red"
                                    
                                    st.markdown(f"<span style='color:{trend_color}'>{trend_text}</span>", unsafe_allow_html=True)
                                
                                fig.update_layout(
                                    title=f"Direction Consistency Trend (Rolling {window_size}-shot std dev)",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title="Direction Standard Deviation (°)",
                                    height=350,
                                    xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Session-by-session accuracy comparison
                    if 'Session' in club_df.columns and club_df['Session'].nunique() > 1:
                        st.subheader("📊 Session-by-Session Accuracy Comparison")
                        
                        session_accuracy = []
                        for session in sorted(club_df['Session'].unique()):
                            session_data = club_df[club_df['Session'] == session]
                            
                            # Skip empty sessions
                            if len(session_data) == 0:
                                continue
                                
                            # Safely get session date
                            if 'Session Date' in session_data.columns and len(session_data) > 0:
                                try:
                                    session_date = session_data['Session Date'].iloc[0]
                                except (IndexError, KeyError):
                                    session_date = f"Session {session}"
                            else:
                                session_date = f"Session {session}"
                            
                            accuracy_metrics = {
                                'Session': session,
                                'Date': session_date,
                                'Shots': len(session_data)
                            }
                            
                            # Only calculate metrics if there's data
                            if 'Carry Deviation Distance' in session_data.columns and session_data['Carry Deviation Distance'].notna().any():
                                accuracy_metrics['Avg Distance Deviation'] = session_data['Carry Deviation Distance'].abs().mean()
                            
                            if 'Launch Direction' in session_data.columns and session_data['Launch Direction'].notna().any():
                                accuracy_metrics['Direction Std Dev'] = session_data['Launch Direction'].std()
                            
                            if 'Club Path' in session_data.columns and session_data['Club Path'].notna().any():
                                accuracy_metrics['Club Path Std Dev'] = session_data['Club Path'].std()
                            
                            session_accuracy.append(accuracy_metrics)
                        
                        session_df = pd.DataFrame(session_accuracy)
                        
                        if len(session_df) > 1:
                            # Show session comparison table
                            st.dataframe(session_df.round(2), use_container_width=True)
                            
                            # Plot session accuracy trends
                            session_cols = st.columns(2)
                            
                            if 'Avg Distance Deviation' in session_df.columns and session_df['Avg Distance Deviation'].notna().any():
                                with session_cols[0]:
                                    fig = px.line(
                                        session_df,
                                        x='Session',
                                        y='Avg Distance Deviation',
                                        title="Distance Accuracy by Session",
                                        markers=True
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            if 'Direction Std Dev' in session_df.columns and session_df['Direction Std Dev'].notna().any():
                                with session_cols[1]:
                                    fig = px.line(
                                        session_df,
                                        x='Session',
                                        y='Direction Std Dev',
                                        title="Direction Consistency by Session",
                                        markers=True
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Performance insights
                            st.subheader("🎯 Performance Insights")
                            insights_cols = st.columns(2)
                            
                            with insights_cols[0]:
                                if 'Avg Distance Deviation' in session_df.columns and len(session_df) >= 2:
                                    try:
                                        # Filter out NaN values for comparison
                                        valid_sessions = session_df.dropna(subset=['Avg Distance Deviation'])
                                        if len(valid_sessions) >= 2:
                                            first_session = valid_sessions.iloc[0]['Avg Distance Deviation']
                                            last_session = valid_sessions.iloc[-1]['Avg Distance Deviation']
                                            improvement = first_session - last_session
                                            
                                            if improvement > 0:
                                                st.success(f"🎉 Distance accuracy improved by {improvement:.1f}m from first to last session!")
                                            elif improvement < 0:
                                                st.warning(f"⚠️ Distance accuracy declined by {abs(improvement):.1f}m from first to last session")
                                            else:
                                                st.info("📊 Distance accuracy remained consistent across sessions")
                                        else:
                                            st.info("📊 Insufficient data for distance accuracy comparison")
                                    except (IndexError, KeyError, ValueError):
                                        st.info("📊 Unable to calculate distance accuracy trend")
                            
                            with insights_cols[1]:
                                if 'Direction Std Dev' in session_df.columns and len(session_df) >= 2:
                                    try:
                                        # Filter out NaN values for comparison
                                        valid_sessions = session_df.dropna(subset=['Direction Std Dev'])
                                        if len(valid_sessions) >= 2:
                                            first_session = valid_sessions.iloc[0]['Direction Std Dev']
                                            last_session = valid_sessions.iloc[-1]['Direction Std Dev']
                                            improvement = first_session - last_session
                                            
                                            if improvement > 0:
                                                st.success(f"🎉 Direction consistency improved by {improvement:.1f}° from first to last session!")
                                            elif improvement < 0:
                                                st.warning(f"⚠️ Direction consistency declined by {abs(improvement):.1f}° from first to last session")
                                            else:
                                                st.info("📊 Direction consistency remained stable across sessions")
                                        else:
                                            st.info("📊 Insufficient data for direction consistency comparison")
                                    except (IndexError, KeyError, ValueError):
                                        st.info("📊 Unable to calculate direction consistency trend")
                        else:
                            st.info("📊 Need at least 2 sessions with data for comparison")
                    
                    # Deviation plots
                    st.subheader("📊 Accuracy Distribution Analysis")
                    deviation_stats = [
                        ('Carry Deviation Distance', 'Distance Deviation (m)'),
                        ('Total Deviation Distance', 'Total Deviation (m)'),
                        ('Launch Direction', 'Launch Direction (°)'),
                        ('Club Path', 'Club Path (°)'),
                    ]
                    
                    cols = st.columns(2)
                    for i, (col, title) in enumerate(deviation_stats):
                        if col in club_df.columns and club_df[col].notna().any():
                            with cols[i % 2]:
                                try:
                                    fig = px.histogram(
                                        club_df, 
                                        x=col, 
                                        title=f"{title} Distribution",
                                        nbins=15
                                    )
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating distribution for {title}: {str(e)}")
                
                with tab3:
                    # Shot pattern analysis
                    if ('Launch Direction' in club_df.columns and 'Carry Distance' in club_df.columns and 
                        club_df['Launch Direction'].notna().any() and club_df['Carry Distance'].notna().any()):
                        try:
                            # Create scatter plot of shot pattern
                            fig = px.scatter(
                                club_df,
                                x='Launch Direction',
                                y='Carry Distance',
                                color='Smash Factor' if 'Smash Factor' in club_df.columns else None,
                                title="Shot Pattern (Launch Direction vs Carry Distance)",
                                labels={
                                    'Launch Direction': 'Launch Direction (°)',
                                    'Carry Distance': 'Carry Distance (m)'
                                }
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Attack angle vs ball flight
                            if 'Attack Angle' in club_df.columns and club_df['Attack Angle'].notna().any():
                                fig2 = px.scatter(
                                    club_df,
                                    x='Attack Angle',
                                    y='Launch Angle',
                                    color='Carry Distance',
                                    title="Attack Angle vs Launch Angle (colored by distance)",
                                    labels={
                                        'Attack Angle': 'Attack Angle (°)',
                                        'Launch Angle': 'Launch Angle (°)'
                                    }
                                )
                                fig2.update_layout(height=400)
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Direction analysis
                            direction_data = club_df['Launch Direction'].apply(
                                lambda x: 'Right' if pd.notna(x) and x > 2 else ('Left' if pd.notna(x) and x < -2 else 'Center')
                            ).value_counts()
                            
                            fig_bar = px.bar(
                                x=direction_data.index,
                                y=direction_data.values,
                                title="Shot Direction Distribution",
                                labels={'x': 'Direction', 'y': 'Number of Shots'}
                            )
                            fig_bar.update_layout(height=400)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating shot pattern analysis: {str(e)}")
                    else:
                        st.info("Launch Direction and Carry Distance data required for shot pattern analysis.")
                    
                    # Shot Shape Trend Analysis
                    if 'Shot Shape' in club_df.columns and len(club_df) > 5:
                        st.subheader("🎯 Shot Shape Trends")
                        
                        # Shot shape over time
                        shape_cols = st.columns(2)
                        
                        with shape_cols[0]:
                            # Create shot shape trend chart
                            shot_shapes_numeric = club_df['Shot Shape'].map({
                                'Hook': -2, 'Draw': -1, 'Straight': 0, 'Fade': 1, 'Slice': 2,
                                'Pull': -1.5, 'Push': 1.5, 'Unknown': 0
                            })
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=club_df['Club Shot Number'],
                                y=shot_shapes_numeric,
                                mode='lines+markers',
                                name='Shot Shape Trend',
                                line=dict(width=2),
                                marker=dict(size=6, color=shot_shapes_numeric, colorscale='RdYlGn_r'),
                                text=club_df['Shot Shape'],
                                hovertemplate='<b>Shot %{x}</b><br>Shape: %{text}<br><extra></extra>'
                            ))
                            
                            fig.update_layout(
                                title="Shot Shape Trend Over Time",
                                xaxis_title=f"{title_suffix} Shot Number",
                                yaxis_title="Shot Shape",
                                height=350,
                                xaxis=get_smart_xaxis_config(len(club_df)),
                                yaxis=dict(
                                    tickvals=[-2, -1.5, -1, 0, 1, 1.5, 2],
                                    ticktext=['Hook', 'Pull', 'Draw', 'Straight', 'Fade', 'Push', 'Slice']
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with shape_cols[1]:
                            # Rolling shot shape consistency
                            if len(club_df) >= 10:
                                window_size = min(10, len(club_df) // 3)
                                
                                # Calculate rolling consistency (how many straight/draw/fade in window)
                                good_shapes = ['Straight', 'Draw', 'Fade']
                                club_df_temp = club_df.copy()
                                club_df_temp['Is_Good_Shape'] = club_df_temp['Shot Shape'].isin(good_shapes).astype(int)
                                club_df_temp['Rolling_Consistency'] = club_df_temp['Is_Good_Shape'].rolling(
                                    window=window_size, min_periods=3
                                ).mean() * 100
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=club_df_temp['Club Shot Number'],
                                    y=club_df_temp['Rolling_Consistency'],
                                    mode='lines+markers',
                                    name='Consistency %',
                                    line=dict(width=2, color='blue'),
                                    marker=dict(size=4)
                                ))
                                
                                # Add target zone
                                fig.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1, 
                                            annotation_text="Target Zone (70%+)", annotation_position="top left")
                                
                                fig.update_layout(
                                    title=f"Shot Shape Consistency (Rolling {window_size}-shot %)",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title="Consistency (%)",
                                    height=350,
                                    xaxis=get_smart_xaxis_config(len(club_df))
                                )
                                st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.subheader("🌪️ Spin Analysis")
                    
                    spin_cols = st.columns(3)
                    if 'Backspin' in club_df.columns and club_df['Backspin'].notna().any():
                        with spin_cols[0]:
                            avg_backspin = club_df['Backspin'].mean()
                            # Golf-specific backspin context
                            if 'Driver' in title_suffix:
                                optimal_backspin = "2000-2500 rpm"
                                spin_quality = "Good" if 2000 <= avg_backspin <= 2500 else "Needs adjustment"
                            elif any(iron in title_suffix for iron in ['7 Iron', '6 Iron', '8 Iron', '9 Iron']):
                                optimal_backspin = "5000-7000 rpm"
                                spin_quality = "Good" if 5000 <= avg_backspin <= 7000 else "Needs adjustment"
                            elif any(wedge in title_suffix for wedge in ['Wedge', 'SW', 'LW', 'GW', 'PW']):
                                optimal_backspin = "8000-12000 rpm"
                                spin_quality = "Good" if 8000 <= avg_backspin <= 12000 else "Needs adjustment"
                            else:
                                optimal_backspin = "Varies by club"
                                spin_quality = ""
                            
                            st.metric("Avg Backspin", f"{avg_backspin:.0f} rpm", 
                                    help=f"Optimal for {title_suffix}: {optimal_backspin}")
                            if spin_quality:
                                color = "green" if spin_quality == "Good" else "orange"
                                st.markdown(f"<span style='color:{color}'>{spin_quality}</span>", unsafe_allow_html=True)
                    
                    if 'Sidespin' in club_df.columns and club_df['Sidespin'].notna().any():
                        with spin_cols[1]:
                            avg_sidespin = club_df['Sidespin'].abs().mean()
                            # Sidespin should generally be low for accuracy
                            sidespin_quality = "Excellent" if avg_sidespin < 300 else ("Good" if avg_sidespin < 500 else "High")
                            color = "green" if sidespin_quality == "Excellent" else ("orange" if sidespin_quality == "Good" else "red")
                            st.metric("Avg Sidespin", f"{avg_sidespin:.0f} rpm", 
                                    help="Lower sidespin = straighter shots")
                            st.markdown(f"<span style='color:{color}'>{sidespin_quality}</span>", unsafe_allow_html=True)
                    
                    if 'Spin Rate' in club_df.columns and club_df['Spin Rate'].notna().any():
                        with spin_cols[2]:
                            avg_total_spin = club_df['Spin Rate'].mean()
                            st.metric("Avg Total Spin", f"{avg_total_spin:.0f} rpm")
                    
                    # Spin vs distance relationship with golf context
                    if ('Backspin' in club_df.columns and 'Carry Distance' in club_df.columns and
                        club_df['Backspin'].notna().any() and club_df['Carry Distance'].notna().any()):
                        
                        st.subheader("🎯 Spin vs Performance Relationship")
                        
                        fig = px.scatter(
                            club_df,
                            x='Backspin',
                            y='Carry Distance',
                            color='Launch Angle' if 'Launch Angle' in club_df.columns else None,
                            title=f"Backspin vs Carry Distance - {title_suffix}",
                            labels={
                                'Backspin': 'Backspin (rpm)',
                                'Carry Distance': 'Carry Distance (m)'
                            }
                        )
                        
                        # Add optimal backspin zones
                        if 'Driver' in title_suffix:
                            fig.add_vrect(x0=2000, x1=2500, fillcolor="green", opacity=0.1, 
                                        annotation_text="Optimal Driver Backspin", annotation_position="top")
                        elif any(iron in title_suffix for iron in ['Iron']):
                            fig.add_vrect(x0=5000, x1=7000, fillcolor="green", opacity=0.1, 
                                        annotation_text="Optimal Iron Backspin", annotation_position="top")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sidespin analysis for shot direction
                    if 'Sidespin' in club_df.columns and club_df['Sidespin'].notna().any():
                        st.subheader("🎯 Shot Direction Analysis")
                        
                        # Create shot shape categories based on sidespin
                        club_df_copy = club_df.copy()
                        club_df_copy['Shot Shape'] = club_df_copy['Sidespin'].apply(
                            lambda x: 'Draw' if pd.notna(x) and x < -200 else 
                                     ('Fade' if pd.notna(x) and x > 200 else 'Straight')
                        )
                        
                        shape_counts = club_df_copy['Shot Shape'].value_counts()
                        
                        fig_shape = px.pie(
                            values=shape_counts.values,
                            names=shape_counts.index,
                            title="Shot Shape Distribution",
                            color_discrete_map={'Straight': 'green', 'Draw': 'blue', 'Fade': 'orange'}
                        )
                        st.plotly_chart(fig_shape, use_container_width=True)
                    
                    # Spin trends
                    st.subheader("📈 Spin Consistency Trends")
                    spin_stats = [
                        ('Backspin', 'Backspin', 'rpm'),
                        ('Sidespin', 'Sidespin', 'rpm'),
                        ('Spin Rate', 'Total Spin Rate', 'rpm'),
                    ]
                    
                    for label, col, unit in spin_stats:
                        if col in club_df.columns and club_df[col].notna().any():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=club_df['Club Shot Number'],
                                y=club_df[col],
                                mode='lines+markers',
                                name=label,
                                line=dict(width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Add consistency bands
                            if col == 'Sidespin':
                                fig.add_hrect(y0=-300, y1=300, fillcolor="green", opacity=0.1, 
                                            annotation_text="Straight Shot Zone", annotation_position="top left")
                            
                            fig.update_layout(
                                title=f"{label} Trend",
                                xaxis_title=f"{title_suffix} Shot Number",
                                yaxis_title=f"{label} ({unit})",
                                height=350,
                                xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab5:
                    st.subheader("�️ Environmental Impact Analysis")
                    
                    # Temperature effects on ball flight
                    if 'Temperature' in club_df.columns and club_df['Temperature'].notna().any():
                        st.subheader("🌡️ Temperature vs Performance")
                        temp_cols = st.columns(2)
                        
                        with temp_cols[0]:
                            avg_temp = club_df['Temperature'].mean()
                            st.metric("Average Temperature", f"{avg_temp:.1f}°C")
                            
                            # Golf-specific temperature insights
                            if avg_temp < 10:
                                temp_effect = "Cold: Ball travels ~2-5% shorter"
                                temp_color = "blue"
                            elif avg_temp > 30:
                                temp_effect = "Hot: Ball travels ~2-5% longer"
                                temp_color = "red"
                            else:
                                temp_effect = "Optimal temperature range"
                                temp_color = "green"
                            
                            st.markdown(f"<span style='color:{temp_color}'>{temp_effect}</span>", 
                                      unsafe_allow_html=True)
                        
                        # Temperature vs distance correlation
                        if 'Carry Distance' in club_df.columns:
                            fig_temp = px.scatter(
                                club_df,
                                x='Temperature',
                                y='Carry Distance',
                                title="Temperature vs Carry Distance",
                                labels={'Temperature': 'Temperature (°C)', 'Carry Distance': 'Carry Distance (m)'},
                                trendline="ols"
                            )
                            
                            # Add temperature effect zones
                            fig_temp.add_vrect(x0=-10, x1=10, fillcolor="blue", opacity=0.1, 
                                             annotation_text="Cold: Shorter carry", annotation_position="top left")
                            fig_temp.add_vrect(x0=30, x1=50, fillcolor="red", opacity=0.1, 
                                             annotation_text="Hot: Longer carry", annotation_position="top right")
                            fig_temp.add_vrect(x0=15, x1=25, fillcolor="green", opacity=0.1, 
                                             annotation_text="Optimal Range", annotation_position="top")
                            
                            st.plotly_chart(fig_temp, use_container_width=True)
                    
                    # Air density and altitude effects
                    air_cols = st.columns(4)
                    env_stats = [
                        ('Temperature', '°C'),
                        ('Air Pressure', 'kPa'),
                        ('Relative Humidity', '%'),
                        ('Air Density', 'g/L')
                    ]
                    
                    for i, (col, unit) in enumerate(env_stats):
                        if col in club_df.columns and club_df[col].notna().any():
                            with air_cols[i]:
                                avg_val = club_df[col].mean()
                                st.metric(f"Avg {col}", f"{avg_val:.1f} {unit}")
                                
                                # Add golf-specific context
                                if col == 'Air Density':
                                    if avg_val < 1.15:
                                        density_effect = "Thin air: Ball travels farther"
                                    elif avg_val > 1.25:
                                        density_effect = "Dense air: Ball travels shorter"
                                    else:
                                        density_effect = "Standard conditions"
                                    st.caption(density_effect)
                                elif col == 'Relative Humidity':
                                    if avg_val > 80:
                                        humidity_effect = "High humidity: Slight distance loss"
                                    else:
                                        humidity_effect = "Good conditions"
                                    st.caption(humidity_effect)
                    
                    # Air density impact with golf context
                    if 'Air Density' in club_df.columns and 'Ball Speed' in club_df.columns:
                        if club_df['Air Density'].notna().any() and club_df['Ball Speed'].notna().any():
                            st.subheader("🌬️ Air Density Impact on Ball Flight")
                            fig = px.scatter(
                                club_df,
                                x='Air Density',
                                y='Ball Speed',
                                title="Air Density vs Ball Speed",
                                labels={
                                    'Air Density': 'Air Density (g/L)',
                                    'Ball Speed': 'Ball Speed (km/h)'
                                }
                            )
                            
                            # Add air density zones
                            fig.add_vrect(x0=1.0, x1=1.15, fillcolor="orange", opacity=0.1, 
                                        annotation_text="Thin Air: More distance", annotation_position="top left")
                            fig.add_vrect(x0=1.15, x1=1.25, fillcolor="green", opacity=0.1, 
                                        annotation_text="Standard Conditions", annotation_position="top")
                            fig.add_vrect(x0=1.25, x1=1.4, fillcolor="blue", opacity=0.1, 
                                        annotation_text="Dense Air: Less distance", annotation_position="top right")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Environmental summary and recommendations
                    st.subheader("📋 Environmental Summary & Golf Tips")
                    
                    recommendations = []
                    
                    # Temperature recommendations
                    if 'Temperature' in club_df.columns and club_df['Temperature'].notna().any():
                        avg_temp = club_df['Temperature'].mean()
                        if avg_temp < 10:
                            recommendations.append("🌡️ **Cold Weather**: Ball flies shorter - consider using one more club")
                        elif avg_temp > 30:
                            recommendations.append("🌡️ **Hot Weather**: Ball flies farther - consider one less club")
                        else:
                            recommendations.append("🌡️ **Optimal Temperature**: Standard ball flight conditions")
                    
                    # Air density recommendations
                    if 'Air Density' in club_df.columns and club_df['Air Density'].notna().any():
                        avg_density = club_df['Air Density'].mean()
                        if avg_density < 1.15:
                            recommendations.append("🏔️ **Thin Air**: Ball carries significantly farther - club down")
                        elif avg_density > 1.25:
                            recommendations.append("🌊 **Dense Air**: Ball carries shorter - club up")
                    
                    # Humidity recommendations
                    if 'Relative Humidity' in club_df.columns and club_df['Relative Humidity'].notna().any():
                        avg_humidity = club_df['Relative Humidity'].mean()
                        if avg_humidity > 80:
                            recommendations.append("💧 **High Humidity**: Slight distance loss, ball may feel heavier")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(rec)
                    else:
                        st.info("Environmental data will help optimize your club selection for different conditions")
    else:
        st.error("No club type data found in the uploaded files.")
else:
    st.info("Please upload one or more CSV files to get started.")
    
    # Show sample data structure
    st.markdown("### Expected CSV Structure")
    st.markdown("Your CSV files should contain columns like:")
    st.code("""
Date, Player, Club Name, Club Type, Club Speed, Ball Speed, 
Smash Factor, Launch Angle, Launch Direction, Carry Distance, 
Total Distance, etc.
""")
