import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Unit conversion functions
def convert_speed_to_usa(kmh_value):
    """Convert km/h to mph"""
    if pd.isna(kmh_value):
        return kmh_value
    return kmh_value * 0.621371

def convert_distance_to_usa(meter_value):
    """Convert meters to yards"""
    if pd.isna(meter_value):
        return meter_value
    return meter_value * 1.09361

def format_speed_metric(value, use_usa_units=False):
    """Format speed value with appropriate units"""
    if pd.isna(value):
        return "N/A"
    if use_usa_units:
        converted_value = convert_speed_to_usa(value)
        return f"{converted_value:.1f} mph"
    else:
        return f"{value:.1f} km/h"

def format_distance_metric(value, use_usa_units=False):
    """Format distance value with appropriate units"""
    if pd.isna(value):
        return "N/A"
    if use_usa_units:
        converted_value = convert_distance_to_usa(value)
        return f"{converted_value:.1f} yd"
    else:
        return f"{value:.1f}m"

def get_speed_unit_label(use_usa_units=False):
    """Get the speed unit label for charts"""
    return "mph" if use_usa_units else "km/h"

def get_distance_unit_label(use_usa_units=False):
    """Get the distance unit label for charts"""
    return "yards" if use_usa_units else "meters"

# Variance and consistency calculation functions
def calculate_rolling_variance(data, window=5):
    """Calculate rolling variance for consistency tracking"""
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data.rolling(window=window, center=True).var()

def calculate_rolling_std(data, window=5):
    """Calculate rolling standard deviation for consistency tracking"""
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    return data.rolling(window=window, center=True).std()

def get_consistency_rating(std_value, metric_type):
    """Get consistency rating based on standard deviation and metric type"""
    if pd.isna(std_value):
        return "Insufficient Data", "⚪"
    
    # Define thresholds for different metric types
    thresholds = {
        'distance': {'excellent': 5, 'good': 10, 'average': 20},  # meters/yards
        'speed': {'excellent': 2, 'good': 5, 'average': 10},      # km/h or mph
        'angle': {'excellent': 2, 'good': 5, 'average': 10},      # degrees
        'smash_factor': {'excellent': 0.05, 'good': 0.1, 'average': 0.2},  # ratio
        'spin': {'excellent': 200, 'good': 500, 'average': 1000}  # rpm
    }
    
    if metric_type not in thresholds:
        metric_type = 'distance'  # default
    
    t = thresholds[metric_type]
    
    if std_value <= t['excellent']:
        return "Excellent", "🔥"
    elif std_value <= t['good']:
        return "Good", "👍"
    elif std_value <= t['average']:
        return "Average", "📊"
    else:
        return "Needs Work", "💪"

def format_variance_display(variance, metric_type, use_usa_units=False):
    """Format variance value for display with appropriate units"""
    if pd.isna(variance):
        return "N/A"
    
    std_dev = np.sqrt(variance)
    
    if metric_type in ['distance']:
        if use_usa_units:
            std_dev = convert_distance_to_usa(std_dev)
            return f"±{std_dev:.1f} yd"
        else:
            return f"±{std_dev:.1f}m"
    elif metric_type in ['speed']:
        if use_usa_units:
            std_dev = convert_speed_to_usa(std_dev)
            return f"±{std_dev:.1f} mph"
        else:
            return f"±{std_dev:.1f} km/h"
    elif metric_type in ['angle']:
        return f"±{std_dev:.1f}°"
    elif metric_type in ['smash_factor']:
        return f"±{std_dev:.3f}"
    elif metric_type in ['spin']:
        return f"±{std_dev:.0f} rpm"
    else:
        return f"±{std_dev:.2f}"

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

# Add file upload tips and Garmin R10 export guide
with st.expander("📁 File Upload Tips & Garmin R10 Export Guide", expanded=False):
    
    # Create tabs for organization
    guide_tab1, guide_tab2 = st.tabs(["🚀 How to Export from Garmin R10", "📋 Upload Tips"])
    
    with guide_tab1:
        st.markdown("""
        ## 🏌️ **Complete Guide: Exporting CSV Files from Your Garmin R10**
        
        ### **📱 Method 1: Garmin Golf App (Recommended)**
        
        #### **Step 1: Connect & Sync**
        1. **Open the Garmin Golf app** on your smartphone
        2. **Ensure your R10 is connected** via Bluetooth
        3. **Complete your practice session** and let data sync automatically
        4. **Verify data appears** in your session history
        
        #### **Step 2: Export Your Data**
        1. **Navigate to 'Sessions'** or 'History' in the app
        2. **Select the session(s)** you want to export
        3. **Look for 'Share' or 'Export' option** (usually three dots menu or share icon)
        4. **Choose 'Export as CSV'** or 'Save Data'
        5. **Select location** to save (email, cloud storage, or local storage)
        
        ---
        
        ### **💻 Method 2: Garmin Connect (Web/Desktop)**
        
        #### **Step 1: Access Garmin Connect**
        1. **Visit** [connect.garmin.com](https://connect.garmin.com)
        2. **Log in** with your Garmin account
        3. **Navigate to 'Activities'** or 'Golf' section
        
        #### **Step 2: Find Your R10 Sessions**
        1. **Filter by date range** when you used your R10
        2. **Look for 'Approach' or 'Golf' activities**
        3. **Click on individual sessions** to view details
        
        #### **Step 3: Export Data**
        1. **Click on a session** to open details
        2. **Look for 'Export' or gear/settings icon**
        3. **Select 'Export Original'** or 'Download CSV'
        4. **Save to your computer**
        
        ---
        
        ### **📊 Method 3: Garmin Express (Desktop Software)**
        
        #### **Step 1: Install & Setup**
        1. **Download Garmin Express** from garmin.com
        2. **Install** on your computer
        3. **Connect your R10** via USB cable
        
        #### **Step 2: Export Process**
        1. **Open Garmin Express**
        2. **Select your R10 device**
        3. **Go to 'Data Export' or 'Backup'**
        4. **Choose date range** for your sessions
        5. **Export as CSV format**
        
        ---
        
        ### **✅ What Your CSV Should Contain:**
        
        **Essential Columns:**
        - `Club Speed` (km/h)
        - `Ball Speed` (km/h) 
        - `Smash Factor`
        - `Carry Distance` (m)
        - `Total Distance` (m)
        - `Launch Angle` (°)
        - `Attack Angle` (°)
        - `Backspin` (rpm)
        - `Sidespin` (rpm)
        - `Launch Direction` (°)
        - `Club Path` (°)
        
        **Additional Helpful Columns:**
        - `Date` & `Time`
        - `Club Type` (Driver, 7 Iron, etc.)
        - `Session` ID or Name
        
        ---
        
        ### **🔧 Troubleshooting Export Issues:**
        
        **❌ Can't Find Export Option?**
        - Update your Garmin Golf app to latest version
        - Try logging out and back in
        - Check if data has fully synced from R10
        
        **❌ Empty or Corrupted Files?**
        - Ensure your R10 session recorded actual shots
        - Try exporting smaller date ranges
        - Check file isn't corrupted during download
        
        **❌ Missing Key Metrics?**
        - Verify your R10 firmware is up to date
        - Some older sessions may have limited data
        - Ensure proper R10 setup during practice
        
        ---
        
        ### **💡 Pro Tips for Better Data:**
        
        **🎯 Setup Tips:**
        - Position R10 **8 feet behind** the ball
        - Ensure **alignment** with target line
        - Use on **level ground** when possible
        - **Calibrate** before each session
        
        **📈 Data Collection:**
        - Take **10+ shots** per session for meaningful analysis
        - Practice with **same club** for consistency tracking
        - **Note conditions** (wind, temperature) in session names
        - Export data **regularly** to avoid loss
        
        ---
        
        ### **🚨 Still Having Issues?**
        
        **Contact Support:**
        - **Garmin Support:** Visit support.garmin.com
        - **Community Forums:** forums.garmin.com
        - **App Issues:** Check app store for updates
        
        **Alternative Methods:**
        - Some third-party golf apps can sync with Garmin
        - Manual shot tracking if export fails
        - Contact dashboard support if file format issues persist
        """)
    
    with guide_tab2:
        st.markdown("""
        ### **📤 Upload Tips for This Dashboard:**
        
        **✅ For Best Results:**
        - Upload CSV files **one at a time** if you experience issues with multiple files
        - Ensure files are properly formatted CSV exports from Garmin devices
        - **File size limit:** 1GB per file (typical R10 sessions are much smaller)
        - **Supported encodings:** UTF-8, Latin-1
        
        **⚠️ Common Issues:**
        - **Empty files** or files with only headers
        - **Corrupted CSV files** from incomplete downloads
        - **Files with non-standard encoding** (try re-exporting)
        - **Missing key columns** (club speed, ball speed, distances)
        
        **🔧 Quick Fixes:**
        - **Re-export** from Garmin if upload fails
        - **Check file size** - very large files may timeout
        - **Verify CSV format** - should open properly in Excel/Google Sheets
        - **Try different export method** if one doesn't work
        
        **📊 What Happens After Upload:**
        - Dashboard automatically detects your data structure
        - **Outlier detection** removes unrealistic shots
        - **Unit conversion** available (metric ↔ imperial)
        - **Multiple sessions** are automatically organized
        """)

uploaded_files = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Validate file sizes first
    oversized_files = []
    for file in uploaded_files:
        file_size_mb = len(file.getvalue()) / (1024 * 1024)
        if file_size_mb > 100:  # 100MB warning threshold
            oversized_files.append(f"{file.name} ({file_size_mb:.1f}MB)")
    
    if oversized_files:
        st.warning(f"⚠️ Large files detected: {', '.join(oversized_files)}")
        st.info("Large files may take longer to process. Consider uploading files individually.")
    
    # Show upload progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load and combine data directly from uploaded files (no temp file saving)
    dfs = []
    successful_files = []
    failed_files = []
    
    try:
        total_files = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing file {i+1} of {total_files}: {file.name}")
            
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Read directly from the uploaded file buffer with error handling
                df_temp = pd.read_csv(file, encoding='utf-8')
                
                # Validate that the file has data
                if df_temp.empty:
                    failed_files.append(f"{file.name} (empty file)")
                    continue
                
                # Validate essential columns exist
                essential_columns = ['Club Type', 'Carry Distance']
                missing_cols = [col for col in essential_columns if col not in df_temp.columns]
                if missing_cols:
                    failed_files.append(f"{file.name} (missing columns: {', '.join(missing_cols)})")
                    continue
                
                # Add source file column
                df_temp['Source File'] = file.name
                dfs.append(df_temp)
                successful_files.append(file.name)
                
            except UnicodeDecodeError:
                # Try different encoding
                try:
                    file.seek(0)
                    df_temp = pd.read_csv(file, encoding='latin-1')
                    if not df_temp.empty:
                        df_temp['Source File'] = file.name
                        dfs.append(df_temp)
                        successful_files.append(file.name)
                    else:
                        failed_files.append(f"{file.name} (empty file)")
                except Exception as e:
                    failed_files.append(f"{file.name} (encoding error: {str(e)[:50]})")
                    
            except pd.errors.EmptyDataError:
                failed_files.append(f"{file.name} (empty or invalid CSV)")
                
            except pd.errors.ParserError as e:
                failed_files.append(f"{file.name} (CSV format error)")
                
            except Exception as e:
                failed_files.append(f"{file.name} (error: {str(e)[:50]})")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Check if we have any successful files
        if not dfs:
            st.error("❌ No files could be processed successfully!")
            if failed_files:
                st.error("Failed files:")
                for failed_file in failed_files:
                    st.write(f"• {failed_file}")
            st.stop()
        
        # Combine successful dataframes
        df = pd.concat(dfs, ignore_index=True)
        
        # Display success/warning messages
        if len(successful_files) == total_files:
            st.success(f"✅ Successfully processed all {len(successful_files)} file(s)")
        else:
            st.warning(f"⚠️ Processed {len(successful_files)} of {total_files} files")
            st.success(f"✅ Successfully processed: {', '.join(successful_files)}")
            if failed_files:
                st.error(f"❌ Failed to process: {', '.join(failed_files)}")
        
        # Show data summary
        st.info(f"📊 Combined dataset: {len(df)} rows, {len(df.columns)} columns from {len(successful_files)} file(s)")
        
    except Exception as e:
        st.error(f"❌ Critical error during file processing: {str(e)}")
        st.error("This might be due to incompatible file formats or corrupted data.")
        st.stop()
    
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
        
        # Find the lowest iron as default selection
        def get_lowest_iron(clubs):
            irons = [club for club in clubs if 'Iron' in str(club)]
            if irons:
                # Extract iron numbers and find the lowest
                iron_numbers = []
                for iron in irons:
                    try:
                        # Extract number from strings like "4 Iron", "7-Iron", etc.
                        import re
                        numbers = re.findall(r'\d+', iron)
                        if numbers:
                            iron_numbers.append((int(numbers[0]), iron))
                    except:
                        continue
                if iron_numbers:
                    iron_numbers.sort()
                    return iron_numbers[0][1]  # Return the iron with lowest number
            # If no irons found, return first club in sorted list
            return clubs[0] if clubs else None
        
        default_club = get_lowest_iron(club_list)
    else:
        club_list = []
        default_club = None
    
    if club_list:
        # Create two columns for club selection and units selection
        col_club, col_units = st.columns([2, 1])
        
        with col_club:
            # Find the index of the default club for the selectbox
            default_index = 0
            if default_club and default_club in club_list:
                default_index = club_list.index(default_club)
            
            selected_club = st.selectbox("Select Club Type", club_list, index=default_index)
        
        with col_units:
            # Units selection
            use_usa_units = st.radio(
                "Units",
                options=[False, True],
                format_func=lambda x: "🇺🇸 USA (mph, yards)" if x else "🌐 Metric (km/h, meters)",
                index=0,
                key="units_selection"
            )
        
        # Filter data by selected club
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
            
            # Show progress for shot shape calculation
            with st.spinner('🎯 Analyzing shot shapes...'):
                # Add shot shape classification using vectorized operations (much faster!)
                if 'Sidespin' in club_df.columns and 'Carry Deviation Angle' in club_df.columns:
                    # Clean the data first
                    sidespin = pd.to_numeric(club_df['Sidespin'], errors='coerce')
                    carry_deviation = pd.to_numeric(club_df['Carry Deviation Angle'], errors='coerce')
                    
                    # Initialize with 'Unknown'
                    shot_shape = pd.Series(['Unknown'] * len(club_df), index=club_df.index)
                    
                    # Create boolean masks for each condition (vectorized operations)
                    valid_data = ~(sidespin.isna() | carry_deviation.isna())
                    
                    # Straight shots
                    straight_mask = valid_data & (sidespin.abs() < 200) & (carry_deviation.abs() < 2)
                    shot_shape[straight_mask] = 'Straight'
                    
                    # Right spin conditions
                    right_spin = valid_data & (sidespin > 0)
                    fade_mask = right_spin & (carry_deviation <= 5) & (carry_deviation >= -2)
                    slice_mask = right_spin & (carry_deviation > 5) & (sidespin > 800)
                    fade_light_mask = right_spin & (carry_deviation > 5) & (sidespin <= 800)
                    pull_mask = right_spin & (carry_deviation < -2)
                    
                    shot_shape[fade_mask] = 'Fade'
                    shot_shape[slice_mask] = 'Slice'
                    shot_shape[fade_light_mask] = 'Fade'
                    shot_shape[pull_mask] = 'Pull'
                    
                    # Left spin conditions  
                    left_spin = valid_data & (sidespin < 0)
                    draw_mask = left_spin & (carry_deviation >= -5) & (carry_deviation <= 2)
                    hook_mask = left_spin & (carry_deviation < -5) & (sidespin < -800)
                    draw_light_mask = left_spin & (carry_deviation < -5) & (sidespin >= -800)
                    push_mask = left_spin & (carry_deviation > 2)
                    
                    shot_shape[draw_mask] = 'Draw'
                    shot_shape[hook_mask] = 'Hook'
                    shot_shape[draw_light_mask] = 'Draw'
                    shot_shape[push_mask] = 'Push'
                    
                    # Assign the vectorized result
                    club_df['Shot Shape'] = shot_shape
                else:
                    # If required columns don't exist, set all to Unknown
                    club_df['Shot Shape'] = 'Unknown'

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
                    st.metric("Avg Carry Distance", format_distance_metric(avg_distance, use_usa_units))
                    if trend_text:
                        st.markdown(f"<span style='color:{trend_color}'>{trend_arrow} {trend_text}</span>", unsafe_allow_html=True)
            
            if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                avg_smash = club_df['Smash Factor'].mean()
                # Club-specific optimal smash factors
                is_driver_only = (club_df['Club'].unique() == ['Driver']).all() if 'Club' in club_df.columns else False
                if is_driver_only:
                    optimal_smash = 1.50
                    help_text = "Optimal smash factor for drivers is around 1.50"
                else:
                    optimal_smash = 1.30
                    help_text = "Optimal smash factor is around 1.30"
                
                delta_smash = avg_smash - optimal_smash
                delta_text = f"({delta_smash:+.3f})" if abs(delta_smash) > 0.001 else ""
                with metrics_cols[1]:
                    st.metric("Avg Smash Factor", f"{avg_smash:.3f}", delta=delta_text, help=help_text)
            
            if 'Club Speed' in club_df.columns and club_df['Club Speed'].notna().any():
                avg_club_speed = club_df['Club Speed'].mean()
                with metrics_cols[2]:
                    st.metric("Avg Club Speed", format_speed_metric(avg_club_speed, use_usa_units))
            
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
                    st.metric("Avg Total Distance", format_distance_metric(avg_total_distance, use_usa_units))
                    if total_text:
                        st.markdown(f"<span style='color:{total_color}'>{total_arrow} {total_text}</span>", unsafe_allow_html=True)
            elif 'Launch Angle' in club_df.columns and club_df['Launch Angle'].notna().any():
                avg_launch = club_df['Launch Angle'].mean()
                with metrics_cols[3]:
                    st.metric("Avg Launch Angle", f"{avg_launch:.1f}°")

            # Performance trends with dynamic units
            trend_stats = [
                ('Club Speed', 'Club Speed', get_speed_unit_label(use_usa_units)),
                ('Ball Speed', 'Ball Speed', get_speed_unit_label(use_usa_units)),
                ('Smash Factor', 'Smash Factor', ''),
                ('Launch Angle', 'Launch Angle', '°'),
                ('Attack Angle', 'Attack Angle', '°'),
                ('Carry Distance', 'Carry Distance', get_distance_unit_label(use_usa_units)),
                ('Total Distance', 'Total Distance', get_distance_unit_label(use_usa_units)),
            ]
            
            # Create subplot layout
            available_stats = [stat for stat in trend_stats if stat[1] in club_df.columns]
            
            # Add smash factor efficiency analysis
            if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                st.subheader("⚡ Smash Factor Efficiency")
                
                # Club-specific optimal smash factors
                is_driver_only = (club_df['Club'].unique() == ['Driver']).all() if 'Club' in club_df.columns else False
                if is_driver_only:
                    optimal_smash = 1.50
                else:
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
            
            # Driver backspin optimization analysis
            if ('Driver' in title_suffix and 'Backspin' in club_df.columns and 'Total Distance' in club_df.columns and 
                'Ball Speed' in club_df.columns and club_df['Backspin'].notna().any()):
                st.subheader("🚀 Driver Distance Optimization")
                
                # Calculate current averages
                avg_backspin = club_df['Backspin'].mean()
                avg_total_distance = club_df['Total Distance'].mean()
                avg_ball_speed = club_df['Ball Speed'].mean()
                
                # Optimal backspin range for drivers (2200-2800 rpm for maximum distance)
                optimal_backspin_low = 2200
                optimal_backspin_high = 2800
                optimal_backspin_target = 2500  # Sweet spot for most drivers
                
                # Distance prediction based on backspin optimization
                # Using approximate formula: for every 500 rpm reduction in excessive backspin, gain ~10-15 yards
                # And for every 500 rpm increase from too-low backspin, gain ~8-12 yards
                
                if avg_backspin > optimal_backspin_high:
                    # Too much backspin - losing distance
                    excess_spin = avg_backspin - optimal_backspin_target
                    # More aggressive gain calculation for high spin (bigger impact)
                    distance_gain_m = (excess_spin / 500) * 12  # ~12m per 500rpm reduction
                    optimization_type = "Reduce Backspin"
                    spin_issue = "Too High"
                    spin_color = "red"
                elif avg_backspin < optimal_backspin_low:
                    # Too little backspin - need more carry
                    deficit_spin = optimal_backspin_target - avg_backspin
                    # More conservative gain for low spin
                    distance_gain_m = (deficit_spin / 500) * 8  # ~8m per 500rpm increase
                    optimization_type = "Increase Backspin"
                    spin_issue = "Too Low"
                    spin_color = "orange"
                else:
                    # Already in optimal range
                    distance_gain_m = 0
                    optimization_type = "Maintain Current"
                    spin_issue = "Optimal"
                    spin_color = "green"
                
                # Convert distance gain to appropriate units
                if use_usa_units:
                    distance_gain_display = distance_gain_m * 1.09361  # Convert to yards
                    distance_unit = "yards"
                    current_distance_display = avg_total_distance * 1.09361
                    predicted_distance_display = (avg_total_distance + distance_gain_m) * 1.09361
                else:
                    distance_gain_display = distance_gain_m
                    distance_unit = "meters"
                    current_distance_display = avg_total_distance
                    predicted_distance_display = avg_total_distance + distance_gain_m
                
                # Display optimization metrics
                opt_cols = st.columns(4)
                
                with opt_cols[0]:
                    st.metric("Current Avg Backspin", f"{avg_backspin:.0f} rpm")
                    st.markdown(f"**Status:** <span style='color:{spin_color}'>{spin_issue}</span>", unsafe_allow_html=True)
                
                with opt_cols[1]:
                    st.metric("Optimal Range", f"{optimal_backspin_low}-{optimal_backspin_high} rpm")
                    st.metric("Target", f"{optimal_backspin_target} rpm")
                
                with opt_cols[2]:
                    st.metric("Current Avg Distance", f"{current_distance_display:.1f} {distance_unit}")
                
                with opt_cols[3]:
                    if distance_gain_m > 2:  # Only show if meaningful gain
                        st.metric("Predicted Optimal Distance", 
                                f"{predicted_distance_display:.1f} {distance_unit}",
                                delta=f"+{distance_gain_display:.1f} {distance_unit}")
                    else:
                        st.metric("Distance Potential", "Already Optimized ✅")
                
                # Provide specific recommendations
                if distance_gain_m > 2:
                    if avg_backspin > optimal_backspin_high:
                        st.warning(f"💡 **Optimization Opportunity:** Your backspin is {avg_backspin - optimal_backspin_target:.0f} rpm above optimal. "
                                 f"Reducing to {optimal_backspin_target} rpm could add **{distance_gain_display:.1f} {distance_unit}** to your drives!")
                        st.info("🔧 **Tips to Reduce Backspin:**\n"
                               "- Tee the ball higher\n"
                               "- Hit up on the ball (positive attack angle)\n"
                               "- Move ball position slightly forward\n"
                               "- Check driver loft (consider lower loft)")
                    elif avg_backspin < optimal_backspin_low:
                        st.warning(f"💡 **Optimization Opportunity:** Your backspin is {optimal_backspin_target - avg_backspin:.0f} rpm below optimal. "
                                 f"Increasing to {optimal_backspin_target} rpm could add **{distance_gain_display:.1f} {distance_unit}** to your drives!")
                        st.info("🔧 **Tips to Increase Backspin:**\n"
                               "- Tee the ball slightly lower\n"
                               "- Reduce attack angle (hit less up)\n"
                               "- Check driver loft (consider higher loft)\n"
                               "- Ensure clean contact on center of face")
                else:
                    st.success("🎉 **Excellent Backspin Control!** Your backspin is already in the optimal range for maximum distance.")
                
                # Show distribution of backspin values
                if len(club_df) > 5:
                    optimal_shots = club_df[(club_df['Backspin'] >= optimal_backspin_low) & 
                                          (club_df['Backspin'] <= optimal_backspin_high)]
                    optimal_percentage = (len(optimal_shots) / len(club_df)) * 100
                    st.info(f"📊 **Consistency:** {optimal_percentage:.1f}% of your drives have optimal backspin ({optimal_backspin_low}-{optimal_backspin_high} rpm)")
            
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
                    
                    # Consistency Analysis Summary
                    st.subheader("📊 Consistency Analysis")
                    
                    consistency_metrics = [
                        ('Carry Distance', 'distance'),
                        ('Club Speed', 'speed'),
                        ('Ball Speed', 'speed'),
                        ('Smash Factor', 'smash_factor'),
                        ('Launch Angle', 'angle'),
                        ('Attack Angle', 'angle')
                    ]
                    
                    consistency_cols = st.columns(3)
                    col_idx = 0
                    
                    for metric_col, metric_type in consistency_metrics:
                        if metric_col in club_df.columns and club_df[metric_col].notna().any():
                            data_std = club_df[metric_col].std()
                            data_var = club_df[metric_col].var()
                            
                            consistency_rating, consistency_icon = get_consistency_rating(data_std, metric_type)
                            variance_display = format_variance_display(data_var, metric_type, use_usa_units)
                            
                            with consistency_cols[col_idx % 3]:
                                st.metric(
                                    f"{metric_col} Consistency",
                                    f"{consistency_rating} {consistency_icon}",
                                    delta=variance_display
                                )
                            
                            col_idx += 1
                    
                    # Overall consistency insights
                    st.subheader("🎯 Key Consistency Insights")
                    
                    insights = []
                    
                    # Check carry distance consistency
                    if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
                        carry_std = club_df['Carry Distance'].std()
                        carry_rating, _ = get_consistency_rating(carry_std, 'distance')
                        if carry_rating == "Excellent":
                            insights.append("🔥 Excellent distance consistency - your swing is very repeatable!")
                        elif carry_rating == "Needs Work":
                            insights.append("💪 Focus on swing consistency to improve distance control")
                    
                    # Check smash factor consistency
                    if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                        smash_std = club_df['Smash Factor'].std()
                        smash_rating, _ = get_consistency_rating(smash_std, 'smash_factor')
                        if smash_rating == "Excellent":
                            insights.append("⚡ Outstanding smash factor consistency - excellent strike quality!")
                        elif smash_rating == "Needs Work":
                            insights.append("🎯 Work on center contact to improve smash factor consistency")
                    
                    # Check direction consistency
                    if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                        direction_std = club_df['Launch Direction'].std()
                        if direction_std <= 8:
                            insights.append("🎯 Excellent directional control - very consistent ball flight!")
                        elif direction_std >= 15:
                            insights.append("📍 Focus on alignment and swing path for better directional consistency")
                    
                    if insights:
                        for insight in insights:
                            st.info(insight)
                    else:
                        st.info("📊 Continue practicing to build more consistency data for analysis")
                    
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
                            # Club-specific optimal ranges
                            if 'Driver' in title_suffix:
                                optimal_shots = club_df[(club_df['Smash Factor'] >= 1.45) & (club_df['Smash Factor'] <= 1.55)]
                            else:
                                optimal_shots = club_df[(club_df['Smash Factor'] >= 1.25) & (club_df['Smash Factor'] <= 1.35)]
                            efficiency_pct = (len(optimal_shots) / len(club_df)) * 100
                            
                            st.metric("Avg Smash Factor", f"{avg_smash:.3f}")
                            st.metric("Optimal Range %", f"{efficiency_pct:.1f}%")
                            
                            # Club-specific ratings
                            if 'Driver' in title_suffix:
                                if avg_smash >= 1.48: rating = "Excellent 🔥"
                                elif avg_smash >= 1.42: rating = "Good 👍"
                                elif avg_smash >= 1.35: rating = "Average 📊"
                                else: rating = "Needs Work 💪"
                            else:
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
                    
                    # Driving Range Shot Dispersion Visualization
                    st.subheader("🏌️ Driving Range Shot Dispersion")
                    
                    # Check if we have the required data for shot dispersion
                    required_dispersion_cols = ['Carry Distance', 'Launch Direction']
                    available_dispersion_cols = [col for col in required_dispersion_cols if col in club_df.columns and club_df[col].notna().any()]
                    
                    # Also check for offline distance which gives lateral dispersion
                    has_offline = 'Offline Distance' in club_df.columns and club_df['Offline Distance'].notna().any()
                    has_carry_deviation = 'Carry Deviation Distance' in club_df.columns and club_df['Carry Deviation Distance'].notna().any()
                    
                    if len(available_dispersion_cols) >= 2 and (has_offline or has_carry_deviation):
                        try:
                            # Create driving range visualization
                            fig_range = go.Figure()
                            
                            # Calculate shot positions
                            carry_distances = club_df['Carry Distance'].dropna()
                            launch_directions = club_df['Launch Direction'].dropna()
                            
                            # Use offline distance if available, otherwise use carry deviation distance
                            if has_offline:
                                lateral_distances = club_df['Offline Distance'].dropna()
                                lateral_col_name = 'Offline Distance'
                            else:
                                lateral_distances = club_df['Carry Deviation Distance'].dropna()
                                lateral_col_name = 'Carry Deviation Distance'
                            
                            # Ensure all arrays have the same length
                            min_length = min(len(carry_distances), len(launch_directions), len(lateral_distances))
                            if min_length > 0:
                                carry_distances = carry_distances.iloc[:min_length]
                                launch_directions = launch_directions.iloc[:min_length]
                                lateral_distances = lateral_distances.iloc[:min_length]
                                
                                # Apply unit conversion if using USA units
                                if use_usa_units:
                                    carry_distances = carry_distances.apply(convert_distance_to_usa)
                                    lateral_distances = lateral_distances.apply(convert_distance_to_usa)
                                
                                # Convert launch direction to radians for calculation
                                import numpy as np
                                launch_rad = np.radians(launch_directions)
                                
                                # Calculate X,Y positions on the range
                                # X = lateral position (left/right)
                                # Y = distance downrange
                                x_positions = lateral_distances  # Lateral dispersion
                                y_positions = carry_distances    # Distance downrange
                                
                                # Color code by smash factor if available
                                if 'Smash Factor' in club_df.columns and len(club_df['Smash Factor'].dropna()) >= min_length:
                                    smash_factors = club_df['Smash Factor'].dropna().iloc[:min_length]
                                    color_data = smash_factors
                                    color_title = "Smash Factor"
                                    colorscale = 'Viridis'
                                    # Set proper scale for smash factor
                                    marker_config = dict(
                                        size=10,
                                        color=color_data,
                                        colorscale=colorscale,
                                        colorbar=dict(
                                            title=color_title,
                                            tickmode="array",
                                            tickvals=[0, 0.5, 1.0, 1.3, 1.5, 1.7],
                                            ticktext=["0", "0.5", "1.0", "1.3", "1.5", "1.7"]
                                        ),
                                        cmin=0,  # Minimum color scale value
                                        cmax=1.7,  # Maximum color scale value
                                        line=dict(width=1, color='black')
                                    )
                                else:
                                    color_data = list(range(min_length))
                                    color_title = "Shot Number"
                                    colorscale = 'Blues'
                                    # Default marker config for shot numbers
                                    marker_config = dict(
                                        size=10,
                                        color=color_data,
                                        colorscale=colorscale,
                                        colorbar=dict(title=color_title),
                                        line=dict(width=1, color='black')
                                    )
                                
                                # Add shot scatter plot
                                fig_range.add_trace(go.Scatter(
                                    x=x_positions,
                                    y=y_positions,
                                    mode='markers',
                                    marker=marker_config,
                                    text=[f'Shot {i+1}<br>Distance: {dist:.1f}{get_distance_unit_label(use_usa_units)}<br>Lateral: {lat:.1f}{get_distance_unit_label(use_usa_units)}<br>Direction: {direction:.1f}°' 
                                          for i, (dist, lat, direction) in enumerate(zip(carry_distances, lateral_distances, launch_directions))],
                                    hovertemplate='%{text}<extra></extra>',
                                    name='Shots'
                                ))
                                
                                # Add target line (centerline)
                                max_distance = max(y_positions) if len(y_positions) > 0 else 200
                                fig_range.add_trace(go.Scatter(
                                    x=[0, 0],
                                    y=[0, max_distance],
                                    mode='lines',
                                    line=dict(color='red', width=3, dash='dash'),
                                    name='Target Line',
                                    hoverinfo='skip'
                                ))
                                
                                # Add range markers every 50m
                                range_markers = list(range(50, int(max_distance) + 50, 50))
                                for distance in range_markers:
                                    if distance <= max_distance:
                                        fig_range.add_trace(go.Scatter(
                                            x=[-50, 50],
                                            y=[distance, distance],
                                            mode='lines',
                                            line=dict(color='lightgray', width=1),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                        # Add distance labels
                                        fig_range.add_annotation(
                                            x=0,
                                            y=distance,
                                            text=f"{distance}m",
                                            showarrow=False,
                                            font=dict(size=10, color='gray'),
                                            xanchor='center'
                                        )
                                
                                # Calculate dispersion statistics
                                lateral_std = np.std(x_positions)
                                distance_std = np.std(y_positions)
                                avg_distance = np.mean(y_positions)
                                avg_lateral = np.mean(x_positions)
                                
                                # Update layout
                                fig_range.update_layout(
                                    title=f"Driving Range Shot Pattern - {title_suffix}",
                                    xaxis_title=f"Lateral Position ({lateral_col_name}) - {get_distance_unit_label(use_usa_units)}",
                                    yaxis_title=f"Carry Distance - {get_distance_unit_label(use_usa_units)}",
                                    xaxis=dict(
                                        range=[-max(abs(min(x_positions)), abs(max(x_positions))) * 1.2,
                                               max(abs(min(x_positions)), abs(max(x_positions))) * 1.2],
                                        zeroline=True,
                                        zerolinecolor='red',
                                        zerolinewidth=2,
                                        showgrid=True,
                                        gridcolor='lightgray'
                                    ),
                                    yaxis=dict(
                                        range=[0, max_distance * 1.1],
                                        showgrid=True,
                                        gridcolor='lightgray'
                                    ),
                                    height=600,
                                    showlegend=True,
                                    legend=dict(x=0.02, y=0.98)
                                )
                                
                                st.plotly_chart(fig_range, use_container_width=True)
                                
                                # Dispersion statistics
                                dispersion_cols = st.columns(4)
                                
                                with dispersion_cols[0]:
                                    # Apply unit conversion to average distance
                                    avg_distance_display = avg_distance
                                    if use_usa_units:
                                        avg_distance_display = avg_distance * 1.09361  # Convert to yards
                                    distance_unit = get_distance_unit_label(use_usa_units)
                                    st.metric("Average Distance", f"{avg_distance_display:.1f}{distance_unit}")
                                    
                                with dispersion_cols[1]:
                                    # Apply unit conversion to distance consistency
                                    distance_std_display = distance_std
                                    if use_usa_units:
                                        distance_std_display = distance_std * 1.09361  # Convert to yards
                                    distance_unit = get_distance_unit_label(use_usa_units)
                                    st.metric("Distance Consistency", f"±{distance_std_display:.1f}{distance_unit}")
                                    
                                with dispersion_cols[2]:
                                    # Apply unit conversion to lateral average
                                    avg_lateral_display = avg_lateral
                                    if use_usa_units:
                                        avg_lateral_display = avg_lateral * 1.09361  # Convert to yards
                                    distance_unit = get_distance_unit_label(use_usa_units)
                                    st.metric("Average Lateral", f"{avg_lateral_display:+.1f}{distance_unit}")
                                    
                                with dispersion_cols[3]:
                                    # Apply unit conversion to lateral spread
                                    lateral_std_display = lateral_std
                                    if use_usa_units:
                                        lateral_std_display = lateral_std * 1.09361  # Convert to yards
                                    distance_unit = get_distance_unit_label(use_usa_units)
                                    st.metric("Lateral Spread", f"±{lateral_std_display:.1f}{distance_unit}")
                                
                                # Dispersion analysis
                                st.markdown("### 📊 Shot Dispersion Analysis")
                                
                                analysis_cols = st.columns(2)
                                
                                with analysis_cols[0]:
                                    # Distance consistency
                                    if distance_std < 5:
                                        st.success(f"🎯 **Excellent distance control** - Your shots are very consistent (±{distance_std:.1f}m)")
                                    elif distance_std < 10:
                                        st.info(f"👍 **Good distance control** - Solid consistency (±{distance_std:.1f}m)")
                                    elif distance_std < 15:
                                        st.warning(f"📊 **Moderate distance control** - Room for improvement (±{distance_std:.1f}m)")
                                    else:
                                        st.error(f"⚠️ **Work on distance control** - High variation (±{distance_std:.1f}m)")
                                
                                with analysis_cols[1]:
                                    # Lateral accuracy
                                    if abs(avg_lateral) < 3 and lateral_std < 8:
                                        st.success(f"🎯 **Excellent accuracy** - Shots centered and tight (±{lateral_std:.1f}m)")
                                    elif abs(avg_lateral) < 8 and lateral_std < 15:
                                        st.info(f"👍 **Good accuracy** - Minor directional tendency ({avg_lateral:+.1f}m)")
                                    elif abs(avg_lateral) < 15 or lateral_std < 20:
                                        st.warning(f"📊 **Moderate accuracy** - Work on direction ({avg_lateral:+.1f}m ±{lateral_std:.1f}m)")
                                    else:
                                        st.error(f"⚠️ **Work on accuracy** - Significant directional issues ({avg_lateral:+.1f}m ±{lateral_std:.1f}m)")
                                
                                # Practice recommendations based on dispersion
                                st.info(f"""
                                **🎯 Dispersion Insights:**
                                - **Your shots cluster around {avg_distance:.0f}m** with ±{distance_std:.1f}m variation
                                - **Lateral bias**: {avg_lateral:+.1f}m ({'right' if avg_lateral > 0 else 'left' if avg_lateral < 0 else 'centered'})
                                - **Shot spread**: ±{lateral_std:.1f}m lateral dispersion
                                - **Target area**: 68% of shots within the displayed pattern
                                """)
                            
                            else:
                                st.warning("Not enough overlapping data points for dispersion visualization")
                                
                        except Exception as e:
                            st.error(f"Error creating driving range visualization: {str(e)}")
                            st.write("Available data columns:", list(club_df.columns))
                    
                    else:
                        missing_cols = [col for col in required_dispersion_cols if col not in club_df.columns]
                        st.info(f"""
                        **Driving Range Visualization Requirements:**
                        - ✅ Available: {', '.join(available_dispersion_cols)}
                        - ❌ Missing: {', '.join(missing_cols)}
                        - ❌ Need lateral data: {'✅' if has_offline or has_carry_deviation else '❌'} (Offline Distance or Carry Deviation Distance)
                        
                        Upload data with Carry Distance, Launch Direction, and Offline Distance to see your shot pattern!
                        """)
                    
                    # Detailed Recommendations
                    st.subheader("🎯 Swing Technique Recommendations")
                    
                    recommendations = []
                    priority_areas = []
                    
                    # Smash Factor Analysis
                    if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
                        avg_smash = club_df['Smash Factor'].mean()
                        # Club-specific targets
                        is_driver_only = (club_df['Club'].unique() == ['Driver']).all() if 'Club' in club_df.columns else False
                        if is_driver_only:
                            optimal_target = 1.50
                            good_range = "1.45-1.55"
                            efficiency_target = "1.50+ for maximum efficiency"
                            low_threshold = 1.35
                            med_threshold = 1.42
                        else:
                            optimal_target = 1.30
                            good_range = "1.25-1.35"
                            efficiency_target = "1.30+ for maximum efficiency"
                            low_threshold = 1.20
                            med_threshold = 1.25
                            
                        if avg_smash < low_threshold:
                            priority_areas.append("Smash Factor")
                            recommendations.append({
                                "icon": "⚡",
                                "title": "Improve Smash Factor (Priority)",
                                "issue": f"Your average smash factor is {avg_smash:.3f}, below optimal range",
                                "solution": "Focus on center face contact. Practice with impact tape or use alignment sticks for consistent setup.",
                                "target": f"Target: {good_range} range"
                            })
                        elif avg_smash < med_threshold:
                            recommendations.append({
                                "icon": "⚡",
                                "title": "Optimize Smash Factor",
                                "issue": f"Your smash factor of {avg_smash:.3f} has room for improvement",
                                "solution": "Work on consistent ball striking. Check your grip pressure and swing tempo.",
                                "target": f"Target: {efficiency_target}"
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
                                
                                # Apply unit conversion for chart data if needed
                                y_data = club_df[col].copy()
                                if use_usa_units:
                                    if col in ['Club Speed', 'Ball Speed']:
                                        y_data = y_data.apply(convert_speed_to_usa)
                                    elif col in ['Carry Distance', 'Total Distance']:
                                        y_data = y_data.apply(convert_distance_to_usa)
                                
                                # Add trend line
                                fig.add_trace(go.Scatter(
                                    x=club_df['Club Shot Number'],
                                    y=y_data,
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
                                
                                # Add variance bands for consistency tracking
                                if len(club_df) > 8:  # Need at least 8 points for meaningful variance
                                    variance_window = min(8, len(club_df) // 2)
                                    rolling_std = calculate_rolling_std(y_data, window=variance_window)
                                    rolling_mean = y_data.rolling(window=variance_window, center=True).mean()
                                    
                                    # Create upper and lower variance bands
                                    upper_band = rolling_mean + rolling_std
                                    lower_band = rolling_mean - rolling_std
                                    
                                    # Add variance bands (±1 standard deviation)
                                    fig.add_trace(go.Scatter(
                                        x=club_df['Club Shot Number'],
                                        y=upper_band,
                                        mode='lines',
                                        name='Consistency Band (+1σ)',
                                        line=dict(width=0),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=club_df['Club Shot Number'],
                                        y=lower_band,
                                        mode='lines',
                                        fill='tonexty',
                                        fillcolor='rgba(128,128,128,0.1)',
                                        name='Consistency Range (±1σ)',
                                        line=dict(width=0),
                                        hovertemplate='<b>Shot %{x}</b><br>' +
                                                    'Consistency Band<br>' +
                                                    '<extra></extra>'
                                    ))
                                    
                                    # Calculate overall consistency metrics
                                    overall_std = y_data.std()
                                    
                                    # Determine metric type for consistency rating
                                    metric_type = 'distance'
                                    if col in ['Club Speed', 'Ball Speed']:
                                        metric_type = 'speed'
                                    elif col in ['Launch Angle', 'Attack Angle']:
                                        metric_type = 'angle'
                                    elif col == 'Smash Factor':
                                        metric_type = 'smash_factor'
                                    elif col in ['Backspin', 'Sidespin']:
                                        metric_type = 'spin'
                                    
                                    consistency_rating, consistency_icon = get_consistency_rating(overall_std, metric_type)
                                    variance_display = format_variance_display(overall_std**2, metric_type, use_usa_units)
                                    
                                    # Add consistency information to chart title
                                    consistency_info = f" | Consistency: {consistency_rating} {consistency_icon} ({variance_display})"
                                else:
                                    consistency_info = " | Need more data for consistency analysis"
                                
                                # Add optimal zones for golf-specific metrics
                                if col == 'Smash Factor':
                                    # Club-specific optimal zones
                                    if 'Driver' in title_suffix:
                                        fig.add_hline(y=1.50, line_dash="dot", line_color="green", 
                                                    annotation_text="Optimal (1.50)", annotation_position="bottom right")
                                        fig.add_hrect(y0=1.45, y1=1.55, fillcolor="green", opacity=0.1, 
                                                    annotation_text="Driver Optimal Range", annotation_position="top left")
                                    else:
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
                                
                                # Add comprehensive trend analysis for all key performance metrics
                                trend_metrics = ['Carry Distance', 'Total Distance', 'Club Speed', 'Ball Speed', 'Smash Factor', 'Launch Angle', 'Attack Angle']
                                if col in trend_metrics and len(club_df) >= 10:
                                    # Calculate trend line using original metric values (before unit conversion)
                                    x_vals = club_df['Club Shot Number']
                                    y_vals_original = club_df[col].dropna()  # Use original values for calculation
                                    x_vals_clean = x_vals[club_df[col].notna()]
                                    
                                    if len(y_vals_original) >= 5:
                                        z = np.polyfit(x_vals_clean, y_vals_original, 1)
                                        trend_line_original = np.poly1d(z)
                                        
                                        # Convert trend line to display units if needed
                                        if use_usa_units and col in ['Club Speed', 'Ball Speed']:
                                            trend_line_display = pd.Series(trend_line_original(x_vals_clean)).apply(convert_speed_to_usa)
                                        elif use_usa_units and col in ['Carry Distance', 'Total Distance']:
                                            trend_line_display = pd.Series(trend_line_original(x_vals_clean)).apply(convert_distance_to_usa)
                                        else:
                                            trend_line_display = trend_line_original(x_vals_clean)
                                        
                                        # Add subtle trend line
                                        fig.add_trace(go.Scatter(
                                            x=x_vals_clean,
                                            y=trend_line_display,
                                            mode='lines',
                                            name='Overall Trend',
                                            line=dict(dash='dot', color='red', width=1),
                                            opacity=0.6
                                        ))
                                        
                                        # Calculate trend direction with metric-specific thresholds
                                        slope = z[0]  # Use original slope for calculations
                                        
                                        # Define improvement thresholds for each metric
                                        if col in ['Carry Distance', 'Total Distance']:
                                            threshold = 1  # 1m per shot
                                            unit_display = "m" if not use_usa_units else "yd"
                                            slope_display = slope if not use_usa_units else slope * 1.09361  # Convert to yards
                                        elif col == 'Club Speed':
                                            threshold = 0.5  # 0.5 km/h per shot
                                            unit_display = "km/h" if not use_usa_units else "mph"
                                            slope_display = slope if not use_usa_units else slope * 0.621371  # Convert to mph
                                        elif col == 'Ball Speed':
                                            threshold = 0.8  # 0.8 km/h per shot
                                            unit_display = "km/h" if not use_usa_units else "mph"
                                            slope_display = slope if not use_usa_units else slope * 0.621371  # Convert to mph
                                        elif col == 'Smash Factor':
                                            threshold = 0.002  # 0.002 per shot
                                            unit_display = ""
                                            slope_display = slope
                                        elif col == 'Launch Angle':
                                            threshold = 0.1  # 0.1° per shot
                                            unit_display = "°"
                                            slope_display = slope
                                        elif col == 'Attack Angle':
                                            # For attack angle, negative slope might be good for irons, positive for driver
                                            threshold = 0.1  # 0.1° per shot
                                            unit_display = "°"
                                            slope_display = slope
                                        
                                        # Determine trend direction and color
                                        if col == 'Attack Angle' and any(iron in title_suffix for iron in ['Iron', '7 Iron', '6 Iron', '8 Iron', '9 Iron']):
                                            # For irons, more negative attack angle is better (hitting down)
                                            if slope < -threshold:
                                                trend_emoji = "📈"
                                                trend_color = "green"
                                            elif slope > threshold:
                                                trend_emoji = "📉"
                                                trend_color = "red"
                                            else:
                                                trend_emoji = "➡️"
                                                trend_color = "blue"
                                        else:
                                            # For most metrics, positive slope is improvement
                                            if slope > threshold:
                                                trend_emoji = "📈"
                                                trend_color = "green"
                                            elif slope < -threshold:
                                                trend_emoji = "📉"
                                                trend_color = "red"
                                            else:
                                                trend_emoji = "➡️"
                                                trend_color = "blue"
                                        
                                        # Add trend annotation with appropriate formatting
                                        if abs(slope_display) >= 0.01 or col in ['Carry Distance', 'Total Distance']:
                                            if col == 'Smash Factor':
                                                trend_text = f"{trend_emoji} {slope_display:+.3f}/shot"
                                            elif col in ['Launch Angle', 'Attack Angle']:
                                                trend_text = f"{trend_emoji} {slope_display:+.2f}{unit_display}/shot"
                                            else:
                                                trend_text = f"{trend_emoji} {slope_display:+.1f}{unit_display}/shot"
                                            
                                            fig.add_annotation(
                                                x=x_vals_clean.iloc[-1],
                                                y=trend_line_display.iloc[-1] if hasattr(trend_line_display, 'iloc') else trend_line_display[-1],
                                                text=trend_text,
                                                showarrow=True,
                                                arrowhead=2,
                                                arrowsize=1,
                                                arrowwidth=2,
                                                arrowcolor=trend_color,
                                                font=dict(color=trend_color, size=12)
                                            )
                                
                                fig.update_layout(
                                    title=f"{label} Trend Over Time{consistency_info}",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title=f"{label} ({unit})" if unit else label,
                                    height=400,
                                    showlegend=True,
                                    xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                                )
                                
                                # Set appropriate y-axis range for Smash Factor
                                if col == 'Smash Factor':
                                    # Use fixed scale from 0 to 1.7 for consistent view across all data
                                    fig.update_yaxes(range=[0, 1.7])
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating chart for {label}: {str(e)}")
                                st.write(f"Data type for {col}: {club_df[col].dtype}")
                                st.write(f"Sample values: {club_df[col].head()}")
                    
                    # 3D Carry Distance vs Smash Factor vs Club Speed Analysis
                    st.subheader("⚡ Carry Distance vs Smash Factor vs Club Speed")
                    
                    # Check if we have all required columns for the 3D chart
                    required_3d_cols = ['Carry Distance', 'Smash Factor', 'Club Speed']
                    available_3d_cols = [col for col in required_3d_cols if col in club_df.columns and club_df[col].notna().any()]
                    
                    if len(available_3d_cols) >= 3:
                        try:
                            # Apply unit conversions for 3D chart data
                            club_speed_3d = club_df['Club Speed'].copy()
                            carry_distance_3d = club_df['Carry Distance'].copy()
                            
                            if use_usa_units:
                                club_speed_3d = club_speed_3d.apply(convert_speed_to_usa)
                                carry_distance_3d = carry_distance_3d.apply(convert_distance_to_usa)
                            
                            # Create 3D scatter plot
                            fig_3d = go.Figure(data=[go.Scatter3d(
                                x=club_speed_3d,
                                y=carry_distance_3d,
                                z=club_df['Smash Factor'],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=club_df['Smash Factor'],
                                    colorscale='Viridis',
                                    colorbar=dict(
                                        title="Smash Factor",
                                        tickmode="array",
                                        tickvals=[0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.7],
                                        ticktext=["0", "0.2", "0.5", "0.8", "1.0", "1.2", "1.3", "1.4", "1.5", "1.7"]
                                    ),
                                    cmin=0,  # Minimum color scale value
                                    cmax=1.7,  # Maximum color scale value
                                    line=dict(width=0.5, color='DarkSlateGrey')
                                ),
                                text=[f'Shot {i+1}<br>Club Speed: {cs:.1f} {get_speed_unit_label(use_usa_units)}<br>Carry Distance: {cd:.1f} {get_distance_unit_label(use_usa_units)}<br>Smash Factor: {sf:.3f}' 
                                      for i, (cs, cd, sf) in enumerate(zip(club_speed_3d, 
                                                                          carry_distance_3d, 
                                                                          club_df['Smash Factor']))],
                                hovertemplate='%{text}<extra></extra>'
                            )])
                            
                            fig_3d.update_layout(
                                title=f"3D Analysis: Carry Distance vs Smash Factor vs Club Speed - {title_suffix}",
                                scene=dict(
                                    xaxis_title=f'Club Speed ({get_speed_unit_label(use_usa_units)})',
                                    yaxis_title=f'Carry Distance ({get_distance_unit_label(use_usa_units)})',
                                    zaxis_title='Smash Factor',
                                    zaxis=dict(
                                        range=[0, 1.7],  # Fixed scale from 0 to 1.7
                                        dtick=0.1  # Tick marks every 0.1
                                    ),
                                    camera=dict(
                                        eye=dict(x=1.5, y=1.5, z=1.5)
                                    )
                                ),
                                height=600,
                                margin=dict(l=0, r=0, b=0, t=40)
                            )
                            
                            # Add a reference plane at the target smash factor of 1.3
                            # Get the range of club speed and carry distance for the plane
                            club_speed_range = [club_df['Club Speed'].min(), club_df['Club Speed'].max()]
                            carry_distance_range = [club_df['Carry Distance'].min(), club_df['Carry Distance'].max()]
                            
                            # Create a mesh for the target plane at Z = 1.3
                            import numpy as np
                            x_plane = np.linspace(club_speed_range[0], club_speed_range[1], 10)
                            y_plane = np.linspace(carry_distance_range[0], carry_distance_range[1], 10)
                            x_mesh, y_mesh = np.meshgrid(x_plane, y_plane)
                            z_mesh = np.full_like(x_mesh, 1.3)  # Target smash factor plane
                            
                            # Add the target plane
                            fig_3d.add_trace(go.Surface(
                                x=x_mesh,
                                y=y_mesh,
                                z=z_mesh,
                                opacity=0.3,
                                colorscale=[[0, 'green'], [1, 'green']],
                                showscale=False,
                                name='Target Zone (1.3)',
                                hovertemplate='Target Smash Factor: 1.3<extra></extra>'
                            ))
                            
                            st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Add insights about the relationship
                            insights_3d_cols = st.columns(2)
                            
                            with insights_3d_cols[0]:
                                # Calculate correlation between club speed and carry distance
                                correlation = club_df['Club Speed'].corr(club_df['Carry Distance'])
                                st.metric("Club Speed ↔ Carry Distance Correlation", f"{correlation:.3f}")
                                
                                if correlation > 0.8:
                                    st.success("🔥 Strong positive correlation - efficient power transfer!")
                                elif correlation > 0.6:
                                    st.info("👍 Good correlation - room for improvement in distance efficiency")
                                else:
                                    st.warning("⚠️ Weak correlation - focus on strike quality and technique")
                            
                            with insights_3d_cols[1]:
                                # Find the optimal combination
                                if 'Smash Factor' in club_df.columns:
                                    max_smash_idx = club_df['Smash Factor'].idxmax()
                                    optimal_shot = club_df.loc[max_smash_idx]
                                    
                                    st.metric("Best Smash Factor Shot", f"{optimal_shot['Smash Factor']:.3f}")
                                    st.caption(f"Club Speed: {optimal_shot['Club Speed']:.1f} km/h")
                                    st.caption(f"Carry Distance: {optimal_shot['Carry Distance']:.1f} m")
                            
                            # Add explanation
                            st.info("""
                            **💡 Understanding the 3D Relationship:**
                            - **Higher Club Speed** should generally produce **longer Carry Distance**
                            - **Smash Factor** shows strike efficiency and affects distance
                            - **Optimal range**: Smash Factor 1.25-1.35 for most clubs
                            - Points higher in the Z-axis (Smash Factor) indicate better strike quality
                            - **Distance** is the ultimate outcome of good club speed and strike efficiency
                            """)
                            
                        except Exception as e:
                            st.error(f"Error creating 3D chart: {str(e)}")
                            st.write("Available columns:", available_3d_cols)
                    else:
                        st.warning(f"3D chart requires Carry Distance, Smash Factor, and Club Speed data. Available: {', '.join(available_3d_cols)}")
                
                with tab2:
                    st.subheader("🎯 Accuracy & Consistency Analysis")
                    
                    # Accuracy metrics
                    accuracy_cols = st.columns(3)
                    
                    # Distance accuracy
                    if 'Carry Deviation Distance' in club_df.columns and club_df['Carry Deviation Distance'].notna().any():
                        with accuracy_cols[0]:
                            avg_deviation = club_df['Carry Deviation Distance'].abs().mean()
                            # Apply unit conversion to average deviation
                            if use_usa_units:
                                avg_deviation = avg_deviation * 1.09361  # Convert to yards
                            distance_unit = get_distance_unit_label(use_usa_units)
                            st.metric("Avg Distance Deviation", f"{avg_deviation:.1f}{distance_unit}")
                    
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
                        
                        # Distance accuracy trend (INVERTED: Lower deviation = better accuracy)
                        if 'Carry Deviation Distance' in club_df.columns and club_df['Carry Deviation Distance'].notna().any():
                            with accuracy_trend_cols[0]:
                                club_df_copy = club_df.copy()
                                # Calculate deviation (lower is better)
                                raw_deviation = club_df_copy['Carry Deviation Distance'].abs().rolling(
                                    window=window_size, center=True
                                ).mean()
                                
                                # Create accuracy score (higher is better) for visualization
                                max_deviation = raw_deviation.max() if raw_deviation.max() > 0 else 10
                                club_df_copy['Distance_Accuracy_Score'] = max_deviation - raw_deviation
                                
                                fig = go.Figure()
                                # Plot accuracy score (higher = better)
                                fig.add_trace(go.Scatter(
                                    x=club_df_copy['Club Shot Number'],
                                    y=club_df_copy['Distance_Accuracy_Score'],
                                    mode='lines+markers',
                                    name='Distance Accuracy Score',
                                    line=dict(width=3, color='green'),  # Green for accuracy (higher is better)
                                    marker=dict(size=4),
                                    hovertemplate='Shot %{x}<br>Accuracy Score: %{y:.1f}<br>Deviation: %{customdata:.1f}m<extra></extra>',
                                    customdata=raw_deviation
                                ))
                                
                                # Add trend line for accuracy score
                                if len(club_df_copy.dropna(subset=['Distance_Accuracy_Score'])) > 5:
                                    x_vals = club_df_copy.dropna(subset=['Distance_Accuracy_Score'])['Club Shot Number']
                                    y_vals = club_df_copy.dropna(subset=['Distance_Accuracy_Score'])['Distance_Accuracy_Score']
                                    raw_vals = club_df_copy.dropna(subset=['Distance_Accuracy_Score'])['Carry Deviation Distance'].abs()
                                    
                                    z = np.polyfit(x_vals, y_vals, 1)
                                    trend_line = np.poly1d(z)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=trend_line(x_vals),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(dash='dash', color='darkgreen', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Calculate improvement (for accuracy score, positive slope = improving)
                                    slope = z[0]
                                    # Convert back to deviation terms for user understanding
                                    avg_deviation_start = raw_vals.iloc[:len(raw_vals)//3].mean()
                                    avg_deviation_end = raw_vals.iloc[-len(raw_vals)//3:].mean()
                                    deviation_change = avg_deviation_end - avg_deviation_start
                                    
                                    if deviation_change < -0.5:  # Deviation decreased = accuracy improved
                                        trend_text = f"📈 Improving: {abs(deviation_change):.1f}m more accurate"
                                        trend_color = "green"
                                    elif deviation_change > 0.5:  # Deviation increased = accuracy declined
                                        trend_text = f"📉 Declining: {deviation_change:.1f}m less accurate"
                                        trend_color = "red"
                                    else:
                                        trend_text = "📊 Consistent accuracy"
                                        trend_color = "blue"
                                    
                                    st.markdown(f"<span style='color:{trend_color}'>{trend_text}</span>", unsafe_allow_html=True)
                                
                                fig.update_layout(
                                    title=f"Distance Accuracy Score Trend (Rolling {window_size}-shot average)",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title="Accuracy Score (Higher = Better)",
                                    height=350,
                                    xaxis=get_smart_xaxis_config(len(club_df))  # Smart axis configuration
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Direction consistency trend (INVERTED: Lower std dev = better consistency)
                        if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
                            with accuracy_trend_cols[1]:
                                club_df_copy = club_df.copy()
                                # Calculate standard deviation (lower is better)
                                raw_std = club_df_copy['Launch Direction'].rolling(
                                    window=window_size, center=True
                                ).std()
                                
                                # Create consistency score (higher is better) for visualization
                                max_std = raw_std.max() if raw_std.max() > 0 else 5
                                club_df_copy['Direction_Consistency_Score'] = max_std - raw_std
                                
                                fig = go.Figure()
                                # Plot consistency score (higher = better)
                                fig.add_trace(go.Scatter(
                                    x=club_df_copy['Club Shot Number'],
                                    y=club_df_copy['Direction_Consistency_Score'],
                                    mode='lines+markers',
                                    name='Direction Consistency Score',
                                    line=dict(width=3, color='green'),  # Green for consistency (higher is better)
                                    marker=dict(size=4),
                                    hovertemplate='Shot %{x}<br>Consistency Score: %{y:.1f}<br>Std Dev: %{customdata:.1f}°<extra></extra>',
                                    customdata=raw_std
                                ))
                                
                                # Add trend line for consistency score
                                if len(club_df_copy.dropna(subset=['Direction_Consistency_Score'])) > 5:
                                    x_vals = club_df_copy.dropna(subset=['Direction_Consistency_Score'])['Club Shot Number']
                                    y_vals = club_df_copy.dropna(subset=['Direction_Consistency_Score'])['Direction_Consistency_Score']
                                    raw_vals = club_df_copy.dropna(subset=['Direction_Consistency_Score'])['Launch Direction']
                                    
                                    z = np.polyfit(x_vals, y_vals, 1)
                                    trend_line = np.poly1d(z)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=trend_line(x_vals),
                                        mode='lines',
                                        name='Trend Line',
                                        line=dict(dash='dash', color='darkgreen', width=2),
                                        opacity=0.8
                                    ))
                                    
                                    # Calculate improvement (for consistency score, positive slope = improving)
                                    slope = z[0]
                                    # Convert back to standard deviation terms for user understanding
                                    std_start = raw_vals.iloc[:len(raw_vals)//3].std()
                                    std_end = raw_vals.iloc[-len(raw_vals)//3:].std()
                                    std_change = std_end - std_start
                                    
                                    if std_change < -0.2:  # Std dev decreased = consistency improved
                                        trend_text = f"📈 Improving: {abs(std_change):.2f}° more consistent"
                                        trend_color = "green"
                                    elif std_change > 0.2:  # Std dev increased = consistency declined
                                        trend_text = f"📉 Declining: {std_change:.2f}° less consistent"
                                        trend_color = "red"
                                    else:
                                        trend_text = "📊 Consistent direction control"
                                        trend_color = "blue"
                                    
                                    st.markdown(f"<span style='color:{trend_color}'>{trend_text}</span>", unsafe_allow_html=True)
                                
                                fig.update_layout(
                                    title=f"Direction Consistency Score Trend (Rolling {window_size}-shot average)",
                                    xaxis_title=f"{title_suffix} Shot Number",
                                    yaxis_title="Consistency Score (Higher = Better)",
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
                    
                    # Enhanced Progress Tracker - Latest vs Previous Sessions
                    if 'Session' in club_df.columns and club_df['Session'].nunique() >= 2:
                        st.subheader("📈 Progress Tracker: Latest Session vs Previous Performance")
                        
                        # Get all sessions sorted
                        all_sessions = sorted(club_df['Session'].unique())
                        latest_session = all_sessions[-1]
                        previous_sessions = all_sessions[:-1]
                        
                        # Get data for latest and previous sessions
                        latest_data = club_df[club_df['Session'] == latest_session]
                        previous_data = club_df[club_df['Session'].isin(previous_sessions)]
                        
                        if len(latest_data) > 0 and len(previous_data) > 0:
                            st.info(f"**Latest Session:** {latest_session} ({len(latest_data)} shots) vs **Previous Sessions:** {len(previous_sessions)} sessions ({len(previous_data)} shots)")
                            
                            # Calculate comprehensive metrics for comparison
                            progress_metrics = []
                            
                            # Define metrics to track with their optimal directions
                            metrics_config = [
                                ('Ball Speed', 'Ball Speed (km/h)', 'higher', '⚡'),
                                ('Club Speed', 'Club Speed (km/h)', 'higher', '🏌️'),
                                ('Smash Factor', 'Smash Factor', 'higher', '💥'),
                                ('Carry Distance', 'Carry Distance (m)', 'higher', '🎯'),
                                ('Launch Direction', 'Launch Direction Consistency (°)', 'lower', '📍'),
                                ('Carry Deviation Distance', 'Distance Accuracy (m)', 'lower', '🎪'),
                                ('Club Path', 'Club Path Consistency (°)', 'lower', '🛤️'),
                                ('Attack Angle', 'Attack Angle Consistency (°)', 'lower', '📐'),
                                ('Backspin', 'Backspin (rpm)', 'stable', '🌪️'),
                                ('Sidespin', 'Sidespin Control (rpm)', 'lower', '↔️')
                            ]
                            
                            for metric, display_name, direction, icon in metrics_config:
                                if metric in club_df.columns and club_df[metric].notna().any():
                                    try:
                                        # Calculate latest session average
                                        if metric in ['Launch Direction', 'Carry Deviation Distance', 'Club Path', 'Attack Angle']:
                                            # For these metrics, we want consistency (lower std dev) or absolute values
                                            if metric == 'Carry Deviation Distance':
                                                latest_avg = latest_data[metric].abs().mean()
                                                previous_avg = previous_data[metric].abs().mean()
                                            elif metric == 'Sidespin':
                                                latest_avg = latest_data[metric].abs().mean()
                                                previous_avg = previous_data[metric].abs().mean()
                                            else:
                                                latest_avg = latest_data[metric].std()
                                                previous_avg = previous_data[metric].std()
                                        else:
                                            # For other metrics, use mean
                                            latest_avg = latest_data[metric].mean()
                                            previous_avg = previous_data[metric].mean()
                                        
                                        # Calculate change
                                        if pd.notna(latest_avg) and pd.notna(previous_avg) and previous_avg != 0:
                                            change = latest_avg - previous_avg
                                            change_pct = (change / previous_avg) * 100
                                            
                                            # Determine if this is an improvement based on metric type
                                            if direction == 'higher':
                                                improved = change > 0
                                            elif direction == 'lower':
                                                improved = change < 0
                                            else:  # stable
                                                improved = abs(change_pct) < 5  # Within 5% is considered stable/good
                                            
                                            progress_metrics.append({
                                                'Metric': display_name,
                                                'Icon': icon,
                                                'Latest': latest_avg,
                                                'Previous': previous_avg,
                                                'Change': change,
                                                'Change %': change_pct,
                                                'Improved': improved,
                                                'Direction': direction
                                            })
                                    except Exception as e:
                                        continue  # Skip metrics that can't be calculated
                            
                            if progress_metrics:
                                # Create progress summary
                                improved_count = sum(1 for m in progress_metrics if m['Improved'])
                                total_count = len(progress_metrics)
                                improvement_rate = (improved_count / total_count) * 100
                                
                                # Overall progress indicator
                                progress_cols = st.columns([1, 2, 1])
                                with progress_cols[1]:
                                    if improvement_rate >= 70:
                                        st.success(f"🔥 **Excellent Progress!** Improved in {improved_count}/{total_count} metrics ({improvement_rate:.0f}%)")
                                    elif improvement_rate >= 50:
                                        st.info(f"👍 **Good Progress!** Improved in {improved_count}/{total_count} metrics ({improvement_rate:.0f}%)")
                                    elif improvement_rate >= 30:
                                        st.warning(f"📊 **Mixed Results** - Improved in {improved_count}/{total_count} metrics ({improvement_rate:.0f}%)")
                                    else:
                                        st.error(f"⚠️ **Focus Needed** - Improved in {improved_count}/{total_count} metrics ({improvement_rate:.0f}%)")
                                
                                # Detailed metrics comparison
                                st.markdown("### 📊 Detailed Metrics Comparison")
                                
                                # Split into improved and needs work
                                improved_metrics = [m for m in progress_metrics if m['Improved']]
                                declined_metrics = [m for m in progress_metrics if not m['Improved']]
                                
                                comparison_cols = st.columns(2)
                                
                                with comparison_cols[0]:
                                    if improved_metrics:
                                        st.markdown("#### ✅ **Improvements**")
                                        for metric in improved_metrics:
                                            with st.expander(f"{metric['Icon']} {metric['Metric']}", expanded=False):
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Latest", f"{metric['Latest']:.2f}")
                                                with col2:
                                                    st.metric("Previous", f"{metric['Previous']:.2f}")
                                                with col3:
                                                    change_sign = "+" if metric['Change'] > 0 else ""
                                                    st.metric("Change", f"{change_sign}{metric['Change']:.2f}", f"{metric['Change %']:+.1f}%")
                                    else:
                                        st.markdown("#### ✅ **Improvements**")
                                        st.info("No improvements detected in this session")
                                
                                with comparison_cols[1]:
                                    if declined_metrics:
                                        st.markdown("#### 🎯 **Areas to Focus On**")
                                        for metric in declined_metrics:
                                            with st.expander(f"{metric['Icon']} {metric['Metric']}", expanded=False):
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Latest", f"{metric['Latest']:.2f}")
                                                with col2:
                                                    st.metric("Previous", f"{metric['Previous']:.2f}")
                                                with col3:
                                                    change_sign = "+" if metric['Change'] > 0 else ""
                                                    st.metric("Change", f"{change_sign}{metric['Change']:.2f}", f"{metric['Change %']:+.1f}%")
                                    else:
                                        st.markdown("#### 🎯 **Areas to Focus On**")
                                        st.success("All metrics improved or stayed stable!")
                                
                                # Progress trend visualization
                                st.markdown("### 📈 Progress Trends")
                                
                                # Create a progress chart showing the key metrics
                                if len(progress_metrics) >= 3:
                                    # Select top metrics for visualization
                                    key_metrics = progress_metrics[:6]  # Show top 6 metrics
                                    
                                    fig_progress = go.Figure()
                                    
                                    # Add bars for each metric
                                    metric_names = [m['Metric'].split('(')[0].strip() for m in key_metrics]
                                    changes = [m['Change %'] for m in key_metrics]
                                    colors = ['green' if m['Improved'] else 'red' for m in key_metrics]
                                    
                                    fig_progress.add_trace(go.Bar(
                                        x=metric_names,
                                        y=changes,
                                        marker_color=colors,
                                        text=[f"{change:+.1f}%" for change in changes],
                                        textposition='auto',
                                        hovertemplate='<b>%{x}</b><br>Change: %{y:.1f}%<extra></extra>'
                                    ))
                                    
                                    fig_progress.update_layout(
                                        title="Latest Session vs Previous Sessions - Key Metrics Change (%)",
                                        xaxis_title="Metrics",
                                        yaxis_title="Change (%)",
                                        height=400,
                                        showlegend=False
                                    )
                                    
                                    # Add a horizontal line at 0%
                                    fig_progress.add_hline(y=0, line_dash="dash", line_color="gray")
                                    
                                    st.plotly_chart(fig_progress, use_container_width=True)
                                
                                # Session progression recommendations
                                st.markdown("### 💡 **Session Recommendations**")
                                
                                if improvement_rate >= 70:
                                    st.success("""
                                    **Keep up the excellent work!** You're showing strong improvement across most metrics.
                                    - Continue your current practice routine
                                    - Focus on maintaining consistency in improved areas
                                    - Consider recording video to analyze your improved technique
                                    """)
                                elif improvement_rate >= 50:
                                    st.info("""
                                    **Good progress overall!** You're improving in most areas.
                                    - Focus extra practice time on the declining metrics
                                    - Consider working with a coach on specific weaknesses
                                    - Keep track of what's working for your improved metrics
                                    """)
                                else:
                                    st.warning("""
                                    **Time to reassess your approach.**
                                    - Consider taking a lesson to identify fundamental issues
                                    - Focus on one or two key metrics rather than trying to fix everything
                                    - Review your setup and basic fundamentals
                                    - Take notes on what feels different in your swing
                                    """)
                            
                            else:
                                st.info("📊 Unable to calculate progress metrics - insufficient data")
                        else:
                            st.info("📊 Need data in both latest and previous sessions for comparison")
                    
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
                                    # Calculate variance metrics for the distribution
                                    data_std = club_df[col].std()
                                    data_var = club_df[col].var()
                                    
                                    # Determine metric type for consistency rating
                                    if 'Distance' in col:
                                        metric_type = 'distance'
                                    elif 'Direction' in col or 'Path' in col:
                                        metric_type = 'angle'
                                    else:
                                        metric_type = 'distance'
                                    
                                    consistency_rating, consistency_icon = get_consistency_rating(data_std, metric_type)
                                    variance_display = format_variance_display(data_var, metric_type, use_usa_units)
                                    
                                    fig = px.histogram(
                                        club_df, 
                                        x=col, 
                                        title=f"{title} Distribution | Variance: {variance_display} | {consistency_rating} {consistency_icon}",
                                        nbins=15
                                    )
                                    
                                    # Add vertical lines for mean and ±1 standard deviation
                                    mean_val = club_df[col].mean()
                                    fig.add_vline(x=mean_val, line_dash="solid", line_color="red", 
                                                annotation_text="Mean", annotation_position="top")
                                    fig.add_vline(x=mean_val + data_std, line_dash="dash", line_color="orange", 
                                                annotation_text="+1σ", annotation_position="top")
                                    fig.add_vline(x=mean_val - data_std, line_dash="dash", line_color="orange", 
                                                annotation_text="-1σ", annotation_position="top")
                                    
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating distribution for {title}: {str(e)}")
                
                with tab3:
                    # Add shot shape reference image
                    try:
                        st.image("assets/shotshape.png", 
                               caption="Shot Shape Reference Guide - Understanding Your Ball Flight Patterns", 
                               width=500)
                        st.markdown("---")  # Add a separator line
                    except Exception:
                        # If image fails to load, continue without it
                        pass
                    
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
    
    # Add Advanced Golf Analytics Section
    st.markdown("---")
    st.header("🎯 Advanced Golf Analytics")
    
    # Advanced Analytics Functions
    def calculate_strokes_gained(df):
        """Calculate strokes gained based on distance to pin"""
        # PGA Tour benchmark distances for strokes gained
        pga_benchmarks = {
            'Driver': {'avg_distance': 285, 'fairway_pct': 0.62},
            'Wood': {'avg_distance': 245, 'fairway_pct': 0.72},
            'Hybrid': {'avg_distance': 220, 'fairway_pct': 0.78},
            'Long Iron': {'avg_distance': 195, 'fairway_pct': 0.82},
            'Mid Iron': {'avg_distance': 165, 'fairway_pct': 0.85},
            'Short Iron': {'avg_distance': 135, 'fairway_pct': 0.88},
            'Wedge': {'avg_distance': 95, 'fairway_pct': 0.92}
        }
        
        strokes_gained = []
        for _, row in df.iterrows():
            club_type = row.get('Club Type', 'Mid Iron')
            carry_distance = row.get('Carry Distance', 0)
            
            if club_type in pga_benchmarks:
                benchmark = pga_benchmarks[club_type]['avg_distance']
                # Simple strokes gained calculation
                sg = (carry_distance - benchmark) / 20  # Rough approximation
                strokes_gained.append(max(-2, min(2, sg)))  # Cap at +/- 2 strokes
            else:
                strokes_gained.append(0)
        
        return strokes_gained

    def analyze_launch_conditions(df):
        """Analyze optimal launch conditions for each club"""
        if 'Launch Angle' not in df.columns or 'Carry Distance' not in df.columns:
            return None
        
        # Optimal launch windows by club type
        optimal_windows = {
            'Driver': {'launch_angle': (10, 14), 'spin_rate': (2000, 2800)},
            'Wood': {'launch_angle': (12, 16), 'spin_rate': (3000, 4000)},
            'Hybrid': {'launch_angle': (14, 18), 'spin_rate': (4000, 5500)},
            'Long Iron': {'launch_angle': (16, 20), 'spin_rate': (5500, 7000)},
            'Mid Iron': {'launch_angle': (18, 22), 'spin_rate': (6500, 8000)},
            'Short Iron': {'launch_angle': (20, 25), 'spin_rate': (7500, 9500)},
            'Wedge': {'launch_angle': (22, 28), 'spin_rate': (8500, 11000)}
        }
        
        results = {}
        for club_type in df['Club Type'].unique():
            if pd.isna(club_type):
                continue
                
            club_data = df[df['Club Type'] == club_type]
            if len(club_data) < 3:
                continue
                
            avg_launch = club_data['Launch Angle'].mean()
            avg_spin = club_data['Spin Rate'].mean() if 'Spin Rate' in club_data.columns else None
            avg_distance = club_data['Carry Distance'].mean()
            
            # Check if in optimal window
            optimal = optimal_windows.get(club_type, {'launch_angle': (15, 20), 'spin_rate': (5000, 8000)})
            launch_optimal = optimal['launch_angle'][0] <= avg_launch <= optimal['launch_angle'][1]
            
            results[club_type] = {
                'avg_launch': avg_launch,
                'avg_spin': avg_spin,
                'avg_distance': avg_distance,
                'launch_optimal': launch_optimal,
                'optimal_range': optimal
            }
        
        return results

    def calculate_dispersion_metrics(df):
        """Calculate shot dispersion and consistency metrics"""
        if 'Carry Deviation Distance' not in df.columns:
            return None
        
        metrics = {}
        for club_type in df['Club Type'].unique():
            if pd.isna(club_type):
                continue
                
            club_data = df[df['Club Type'] == club_type]
            if len(club_data) < 5:
                continue
                
            # Calculate dispersion metrics
            lateral_std = club_data['Carry Deviation Distance'].std()
            distance_std = club_data['Carry Distance'].std()
            
            # Consistency score (lower is better)
            consistency_score = (lateral_std + distance_std) / 2
            
            metrics[club_type] = {
                'lateral_dispersion': lateral_std,
                'distance_dispersion': distance_std,
                'consistency_score': consistency_score,
                'shot_count': len(club_data)
            }
        
        return metrics
    
    # Strokes Gained Analysis
    with st.expander("📊 Strokes Gained Analysis", expanded=False):
        st.markdown("""
        **Strokes Gained** measures your performance relative to PGA Tour benchmarks.
        Positive values mean you're performing better than tour average, negative means room for improvement.
        """)
        
        strokes_gained = calculate_strokes_gained(df)
        df_sg = df.copy()
        df_sg['Strokes Gained'] = strokes_gained
        
        # Strokes gained by club
        sg_by_club = df_sg.groupby('Club Type')['Strokes Gained'].agg(['mean', 'count']).reset_index()
        sg_by_club = sg_by_club[sg_by_club['count'] >= 3]  # Only clubs with 3+ shots
        
        if not sg_by_club.empty:
            fig_sg = px.bar(
                sg_by_club, 
                x='Club Type', 
                y='mean',
                title="Average Strokes Gained by Club Type",
                color='mean',
                color_continuous_scale=['red', 'yellow', 'green'],
                labels={'mean': 'Avg Strokes Gained'}
            )
            fig_sg.add_hline(y=0, line_dash="dash", line_color="black", 
                           annotation_text="Tour Average")
            st.plotly_chart(fig_sg, use_container_width=True)
            
            # Summary insights
            best_club = sg_by_club.loc[sg_by_club['mean'].idxmax(), 'Club Type']
            worst_club = sg_by_club.loc[sg_by_club['mean'].idxmin(), 'Club Type']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Performing Club", best_club, 
                         f"+{sg_by_club['mean'].max():.2f} strokes")
            with col2:
                st.metric("Improvement Opportunity", worst_club, 
                         f"{sg_by_club['mean'].min():.2f} strokes")
    
    # Launch Condition Optimization
    with st.expander("🚀 Launch Condition Optimization", expanded=False):
        st.markdown("""
        **Launch Conditions** determine ball flight and distance. Each club has an optimal launch window
        for maximum efficiency and distance.
        """)
        
        launch_analysis = analyze_launch_conditions(df)
        
        if launch_analysis:
            # Create optimization chart
            optimization_data = []
            for club, data in launch_analysis.items():
                optimization_data.append({
                    'Club Type': club,
                    'Current Launch Angle': data['avg_launch'],
                    'Optimal Min': data['optimal_range']['launch_angle'][0],
                    'Optimal Max': data['optimal_range']['launch_angle'][1],
                    'Distance': data['avg_distance'],
                    'In Window': '✅' if data['launch_optimal'] else '❌'
                })
            
            opt_df = pd.DataFrame(optimization_data)
            
            if not opt_df.empty:
                # Launch angle optimization chart
                fig_opt = go.Figure()
                
                # Add optimal ranges as rectangles
                for _, row in opt_df.iterrows():
                    fig_opt.add_shape(
                        type="rect",
                        x0=row['Club Type'], x1=row['Club Type'],
                        y0=row['Optimal Min'], y1=row['Optimal Max'],
                        fillcolor="lightgreen", opacity=0.3,
                        line=dict(width=0)
                    )
                
                # Add current launch angles
                fig_opt.add_trace(go.Scatter(
                    x=opt_df['Club Type'],
                    y=opt_df['Current Launch Angle'],
                    mode='markers',
                    marker=dict(size=12, color=opt_df['Distance'], 
                              colorscale='viridis', showscale=True,
                              colorbar=dict(title="Distance (m)")),
                    text=opt_df['In Window'],
                    name='Current Launch Angle'
                ))
                
                fig_opt.update_layout(
                    title="Launch Angle Optimization by Club",
                    xaxis_title="Club Type",
                    yaxis_title="Launch Angle (degrees)",
                    showlegend=True
                )
                
                st.plotly_chart(fig_opt, use_container_width=True)
                
                # Optimization recommendations
                st.subheader("🎯 Launch Condition Optimization")
                
                in_window = opt_df[opt_df['In Window'] == '✅']
                out_window = opt_df[opt_df['In Window'] == '❌']
                
                if not in_window.empty:
                    st.success(f"✅ **Clubs in optimal window:** {', '.join(in_window['Club Type'])}")
                
                if not out_window.empty:
                    st.warning(f"❌ **Clubs needing adjustment:** {', '.join(out_window['Club Type'])}")
                    
                    for _, row in out_window.iterrows():
                        current = row['Current Launch Angle']
                        optimal_min = row['Optimal Min']
                        optimal_max = row['Optimal Max']
                        
                        if current < optimal_min:
                            advice = f"Increase launch angle by {optimal_min - current:.1f}°"
                        else:
                            advice = f"Decrease launch angle by {current - optimal_max:.1f}°"
                        
                        st.info(f"**{row['Club Type']}:** {advice}")
    
    # Club Performance Summary
    with st.expander("🏆 Club Performance Summary", expanded=True):
        st.markdown("""
        **Your Best Clubs** - Analysis across all your clubs to identify strengths and preferences.
        """)
        
        if 'Club Type' in df.columns and df['Club Type'].nunique() > 1:
            club_summary = []
            
            for club_type in df['Club Type'].unique():
                if pd.isna(club_type):
                    continue
                    
                club_data = df[df['Club Type'] == club_type]
                if len(club_data) < 3:  # Need at least 3 shots for meaningful analysis
                    continue
                
                # Calculate key metrics
                metrics = {
                    'Club Type': club_type,
                    'Shot Count': len(club_data)
                }
                
                # Distance performance
                if 'Carry Distance' in club_data.columns and club_data['Carry Distance'].notna().any():
                    metrics['Avg Distance'] = club_data['Carry Distance'].mean()
                    metrics['Distance Consistency'] = club_data['Carry Distance'].std()
                
                # Speed performance
                if 'Club Speed' in club_data.columns and club_data['Club Speed'].notna().any():
                    metrics['Avg Club Speed'] = club_data['Club Speed'].mean()
                    
                if 'Ball Speed' in club_data.columns and club_data['Ball Speed'].notna().any():
                    metrics['Avg Ball Speed'] = club_data['Ball Speed'].mean()
                
                # Direction accuracy
                if 'Launch Direction' in club_data.columns and club_data['Launch Direction'].notna().any():
                    metrics['Direction Accuracy'] = club_data['Launch Direction'].std()
                    straight_shots = club_data[club_data['Launch Direction'].abs() <= 5]
                    metrics['Straight Shot %'] = (len(straight_shots) / len(club_data)) * 100
                
                # Strike efficiency
                if 'Smash Factor' in club_data.columns and club_data['Smash Factor'].notna().any():
                    metrics['Avg Smash Factor'] = club_data['Smash Factor'].mean()
                    optimal_shots = club_data[(club_data['Smash Factor'] >= 1.25) & (club_data['Smash Factor'] <= 1.35)]
                    metrics['Optimal Strike %'] = (len(optimal_shots) / len(club_data)) * 100
                
                club_summary.append(metrics)
            
            if club_summary:
                club_summary_df = pd.DataFrame(club_summary)
                
                # Create summary metrics
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    st.markdown("#### 🎯 **Most Consistent Club**")
                    if 'Distance Consistency' in club_summary_df.columns:
                        most_consistent = club_summary_df.loc[club_summary_df['Distance Consistency'].idxmin()]
                        st.metric("Club", most_consistent['Club Type'])
                        # Apply unit conversion to distance consistency
                        consistency_value = most_consistent['Distance Consistency']
                        if use_usa_units:
                            consistency_value = consistency_value * 1.09361  # Convert to yards
                        distance_unit = get_distance_unit_label(use_usa_units)
                        st.metric("Distance Spread", f"±{consistency_value:.1f}{distance_unit}")
                        if 'Shot Count' in most_consistent:
                            st.caption(f"Based on {most_consistent['Shot Count']} shots")
                        st.success("💡 **Use this club** when you need predictable distance control!")
                
                with summary_cols[1]:
                    st.markdown("#### 🚀 **Best Distance Club**")
                    if 'Avg Distance' in club_summary_df.columns:
                        # Find best distance relative to club type (avoid comparing driver to wedge)
                        # For now, just show absolute best, but could be improved with club-type weighting
                        best_distance = club_summary_df.loc[club_summary_df['Avg Distance'].idxmax()]
                        st.metric("Club", best_distance['Club Type'])
                        # Apply unit conversion to average distance
                        avg_distance_value = best_distance['Avg Distance']
                        if use_usa_units:
                            avg_distance_value = avg_distance_value * 1.09361  # Convert to yards
                        distance_unit = get_distance_unit_label(use_usa_units)
                        st.metric("Avg Carry", f"{avg_distance_value:.1f}{distance_unit}")
                        if 'Shot Count' in best_distance:
                            st.caption(f"Based on {best_distance['Shot Count']} shots")
                        st.success("💡 **This club** gives you maximum distance!")
                
                with summary_cols[2]:
                    st.markdown("#### 🎯 **Most Accurate Club**")
                    if 'Direction Accuracy' in club_summary_df.columns:
                        most_accurate = club_summary_df.loc[club_summary_df['Direction Accuracy'].idxmin()]
                        st.metric("Club", most_accurate['Club Type'])
                        st.metric("Direction Spread", f"±{most_accurate['Direction Accuracy']:.1f}°")
                        if 'Straight Shot %' in most_accurate:
                            st.metric("Straight Shots", f"{most_accurate['Straight Shot %']:.1f}%")
                        if 'Shot Count' in most_accurate:
                            st.caption(f"Based on {most_accurate['Shot Count']} shots")
                        st.success("💡 **Go with this club** when accuracy is critical!")
                
                # Additional insights
                st.markdown("---")
                st.markdown("#### 📊 **Club Performance Overview**")
                
                # Create a comprehensive table with unit conversions
                display_df = club_summary_df.copy()
                
                # Get unit labels
                distance_unit = get_distance_unit_label(use_usa_units)
                speed_unit = get_speed_unit_label(use_usa_units)
                
                # Apply unit conversions and update column names
                if 'Avg Distance' in display_df.columns:
                    if use_usa_units:
                        display_df['Avg Distance'] = display_df['Avg Distance'] * 1.09361  # Convert to yards
                    display_df = display_df.rename(columns={'Avg Distance': f'Avg Distance ({distance_unit})'})
                    display_df[f'Avg Distance ({distance_unit})'] = display_df[f'Avg Distance ({distance_unit})'].round(1)
                
                if 'Distance Consistency' in display_df.columns:
                    if use_usa_units:
                        display_df['Distance Consistency'] = display_df['Distance Consistency'] * 1.09361  # Convert to yards
                    display_df = display_df.rename(columns={'Distance Consistency': f'Distance Consistency (±{distance_unit})'})
                    display_df[f'Distance Consistency (±{distance_unit})'] = display_df[f'Distance Consistency (±{distance_unit})'].round(1)
                
                if 'Avg Club Speed' in display_df.columns:
                    if use_usa_units:
                        display_df['Avg Club Speed'] = display_df['Avg Club Speed'] * 0.621371  # Convert to mph
                    display_df = display_df.rename(columns={'Avg Club Speed': f'Avg Club Speed ({speed_unit})'})
                    display_df[f'Avg Club Speed ({speed_unit})'] = display_df[f'Avg Club Speed ({speed_unit})'].round(1)
                
                if 'Avg Ball Speed' in display_df.columns:
                    if use_usa_units:
                        display_df['Avg Ball Speed'] = display_df['Avg Ball Speed'] * 0.621371  # Convert to mph
                    display_df = display_df.rename(columns={'Avg Ball Speed': f'Avg Ball Speed ({speed_unit})'})
                    display_df[f'Avg Ball Speed ({speed_unit})'] = display_df[f'Avg Ball Speed ({speed_unit})'].round(1)
                
                # Round other columns
                if 'Direction Accuracy' in display_df.columns:
                    display_df['Direction Accuracy (±°)'] = display_df['Direction Accuracy'].round(1)
                    display_df = display_df.drop(columns=['Direction Accuracy'])
                    
                if 'Straight Shot %' in display_df.columns:
                    display_df['Straight Shot %'] = display_df['Straight Shot %'].round(1)
                    
                if 'Avg Smash Factor' in display_df.columns:
                    display_df['Avg Smash Factor'] = display_df['Avg Smash Factor'].round(3)
                    
                if 'Optimal Strike %' in display_df.columns:
                    display_df['Optimal Strike %'] = display_df['Optimal Strike %'].round(1)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Performance recommendations
                st.markdown("#### 💡 **Personalized Recommendations**")
                
                recommendations = []
                
                # Find clubs that need work
                if 'Direction Accuracy' in club_summary_df.columns:
                    worst_accuracy = club_summary_df.loc[club_summary_df['Direction Accuracy'].idxmax()]
                    if worst_accuracy['Direction Accuracy'] > 15:
                        recommendations.append(
                            f"🎯 **Work on {worst_accuracy['Club Type']} accuracy** - Direction spread of ±{worst_accuracy['Direction Accuracy']:.1f}° suggests alignment practice needed"
                        )
                
                if 'Distance Consistency' in club_summary_df.columns:
                    worst_consistency = club_summary_df.loc[club_summary_df['Distance Consistency'].idxmax()]
                    if worst_consistency['Distance Consistency'] > 20:
                        # Apply unit conversion for the recommendation text
                        consistency_value = worst_consistency['Distance Consistency']
                        distance_unit = "m"
                        if use_usa_units:
                            consistency_value = consistency_value * 1.09361  # Convert to yards
                            distance_unit = "yd"
                        recommendations.append(
                            f"📏 **Improve {worst_consistency['Club Type']} consistency** - Distance spread of ±{consistency_value:.1f}{distance_unit} indicates tempo/contact work needed"
                        )
                
                if 'Avg Smash Factor' in club_summary_df.columns:
                    low_smash = club_summary_df[club_summary_df['Avg Smash Factor'] < 1.20]
                    if not low_smash.empty:
                        worst_smash = low_smash.loc[low_smash['Avg Smash Factor'].idxmin()]
                        recommendations.append(
                            f"⚡ **Focus on {worst_smash['Club Type']} strike quality** - Smash factor of {worst_smash['Avg Smash Factor']:.3f} suggests contact improvement needed"
                        )
                
                if recommendations:
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        st.warning(rec)
                else:
                    st.success("🎉 **Excellent performance across all clubs!** Your consistency and accuracy are solid.")
        
        else:
            st.info("Upload data with multiple club types to see your club performance summary.")

    # Shot Dispersion Analysis
    with st.expander("🎯 Shot Dispersion & Consistency", expanded=False):
        st.markdown("""
        **Shot Dispersion** measures how consistent your shots are. Lower dispersion = more consistent.
        This helps identify which clubs you can trust and which need more practice.
        """)
        
        dispersion_metrics = calculate_dispersion_metrics(df)
        
        if dispersion_metrics:
            # Create dispersion comparison
            disp_data = []
            for club, metrics in dispersion_metrics.items():
                # Apply unit conversion for dispersion metrics
                lateral_disp = metrics['lateral_dispersion']
                distance_disp = metrics['distance_dispersion']
                if use_usa_units:
                    lateral_disp = lateral_disp * 1.09361  # Convert to yards
                    distance_disp = distance_disp * 1.09361  # Convert to yards
                
                distance_unit = get_distance_unit_label(use_usa_units)
                disp_data.append({
                    'Club Type': club,
                    f'Lateral Dispersion ({distance_unit})': lateral_disp,
                    f'Distance Dispersion ({distance_unit})': distance_disp,
                    'Consistency Score': metrics['consistency_score'],
                    'Shot Count': metrics['shot_count']
                })
            
            disp_df = pd.DataFrame(disp_data)
            
            if not disp_df.empty:
                # Consistency ranking
                disp_df_sorted = disp_df.sort_values('Consistency Score')
                
                distance_unit = get_distance_unit_label(use_usa_units)
                fig_disp = px.scatter(
                    disp_df,
                    x=f'Lateral Dispersion ({distance_unit})',
                    y=f'Distance Dispersion ({distance_unit})',
                    size='Shot Count',
                    color='Consistency Score',
                    hover_name='Club Type',
                    title="Shot Dispersion by Club Type",
                    color_continuous_scale='RdYlGn_r'  # Red = inconsistent, Green = consistent
                )
                
                st.plotly_chart(fig_disp, use_container_width=True)
                
                # Consistency rankings
                st.subheader("📊 Consistency Rankings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Most Consistent Clubs:**")
                    for i, (_, row) in enumerate(disp_df_sorted.head(3).iterrows(), 1):
                        st.write(f"{i}. {row['Club Type']} (Score: {row['Consistency Score']:.1f})")
                
                with col2:
                    st.markdown("**Clubs Needing Practice:**")
                    for i, (_, row) in enumerate(disp_df_sorted.tail(3).iterrows(), 1):
                        st.write(f"{i}. {row['Club Type']} (Score: {row['Consistency Score']:.1f})")

else:
    st.error("No club type data found in the uploaded files.")

# If no files uploaded, show instructions
if not uploaded_files:
    st.info("Please upload one or more CSV files to get started.")
    
    # Show sample data structure
    st.markdown("### Expected CSV Structure")
    st.markdown("Your CSV files should contain columns like:")
    st.code("""
Date, Player, Club Name, Club Type, Club Speed, Ball Speed, 
Smash Factor, Launch Angle, Launch Direction, Carry Distance, 
Total Distance, etc.
""")
