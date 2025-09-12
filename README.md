# 🏌️ Garmin R10 Golf Analytics Dashboard

A comprehensive golf performance analytics dashboard built with Streamlit for analyzing data from the Garmin Approach R10 portable launch monitor. Transform your practice sessions into actionable insights to improve your golf game.

## 📊 Features Overview

### 🎯 **Multi-File Upload & Processing**
- Upload multiple CSV files from your Garmin R10 sessions
- Automatic data encoding detection (UTF-8/Latin-1)
- Progress bars and error handling for robust file processing
- Session-based organization for tracking progress over time

### 📈 **Performance Analytics**

#### **Club Performance Analysis**
- Individual club analysis with detailed metrics
- Key performance indicators (KPIs) for each club
- Ball speed, club speed, smash factor, and carry distance tracking
- Launch angle, attack angle, and spin rate analysis

#### **3D Visualization**
- Interactive 3D scatter plot showing Club Speed vs Carry Distance vs Smash Factor
- Color-coded efficiency visualization with 0-1.7 smash factor scale
- Target reference plane at optimal 1.3 smash factor
- Hover details for individual shot analysis

#### **Trend Analysis**
- Rolling average performance trends over practice sessions
- Smart trend detection with improvement/decline indicators
- Customizable moving averages based on session length
- Visual trend lines with statistical analysis

### 🎪 **Accuracy & Consistency Analysis**

#### **Shot Dispersion Visualization**
- Driving range shot pattern visualization (top-down view)
- Interactive scatter plot showing where shots landed
- Distance markers and target line reference
- Lateral and distance accuracy statistics

#### **Consistency Scoring**
- **Distance Accuracy Score**: Inverted visualization where higher = better accuracy
- **Direction Consistency Score**: Inverted visualization where higher = better consistency
- Proper "lower is better" metric handling for accuracy metrics
- Color-coded performance with green indicating improvement

#### **Session Comparison**
- Session-by-session accuracy comparison
- Statistical analysis of improvement trends
- Visual progress tracking across multiple practice sessions

### 🎯 **Shot Shape Analysis**
- Shot shape distribution pie charts
- Comprehensive breakdown by shot type (Straight, Draw, Fade, Hook, Slice, etc.)
- Shot shape consistency ratings and recommendations
- Color-coded visualization for different shot patterns

### 📊 **Advanced Analytics**

#### **Strokes Gained Analysis**
- Distance-based performance vs tour averages
- Quantified improvement opportunities
- Benchmarking against professional standards

#### **Launch Condition Optimization**
- Optimal launch angle recommendations
- Spin rate analysis for carry distance maximization
- Attack angle optimization guidance

#### **Progress Tracker**
- Latest session vs previous sessions comparison
- 10 key golf metrics tracking with smart direction handling
- Improvement rate calculation and color-coded feedback
- Detailed side-by-side comparison of improvements vs areas needing work

### 🎯 **Personalized Recommendations**

#### **Swing Technique Recommendations**
- Smash factor improvement suggestions
- Shot shape correction guidance
- Attack angle optimization tips
- Club path consistency recommendations

#### **Launch Condition Optimization**
- Specific launch angle adjustments
- Spin rate optimization strategies
- Equipment and setup recommendations

#### **Session-Based Insights**
- Performance trend analysis
- Practice focus area identification
- Improvement rate assessment with actionable advice

## 🚀 **Getting Started**

### **Using Docker (Recommended)**

```bash
# Run the pre-built image from GitHub Container Registry
docker run -d -p 8501:8501 -v $(pwd)/data:/app/data ghcr.io/rpetersson/garmin-r10-dashboard:latest

# Or use docker-compose
docker-compose up -d
```

### **Using Kubernetes**

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Get the service URL
kubectl get service garmin-dashboard-service
```

### **Local Development**

```bash
# Clone the repository
git clone https://github.com/rpetersson/garmin-r10-dashboard.git
cd garmin-r10-dashboard

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🐳 **Docker Commands**

### **Essential Commands**
```bash
# Build locally
docker build -t garmin-dashboard .

# Run locally built image
docker run -d -p 8501:8501 -v $(pwd)/data:/app/data garmin-dashboard

# Pull latest from GitHub
docker pull ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

### **GitHub Actions CI/CD**
- Automated Docker image building
- Security scanning with Trivy
- Multi-platform support (linux/amd64, linux/arm64)
- Automatic publishing to GitHub Container Registry

## 📋 **Data Requirements**

### **Supported Columns**
The app automatically detects and analyzes available data columns:

- **Basic Metrics**: Ball Speed, Club Speed, Smash Factor, Carry Distance
- **Launch Data**: Launch Angle, Launch Direction, Attack Angle
- **Spin Data**: Backspin, Sidespin, Spin Axis
- **Accuracy Data**: Offline Distance, Carry Deviation Distance
- **Club Data**: Club Type, Club Shot Number
- **Session Data**: Session identifier for multi-session analysis

### **File Format**
- CSV files exported from Garmin R10
- Automatic header detection
- Multiple encoding support (UTF-8, Latin-1)
- Flexible column naming recognition

## 📊 **Dashboard Sections**

### **1. Performance Trends Tab**
- Key performance indicators overview
- Individual metric trend charts
- 3D Club Speed vs Carry Distance vs Smash Factor visualization
- Statistical trend analysis

### **2. Accuracy Analysis Tab**
- Driving range shot dispersion visualization
- Distance and direction accuracy scores
- Consistency trend analysis
- Session-by-session comparison

### **3. Advanced Analytics Tab**
- Strokes gained analysis
- Launch condition optimization
- Spin rate and attack angle analysis
- Equipment recommendations

### **4. Progress Tracker**
- Latest vs previous session comparison
- Comprehensive improvement metrics
- Visual progress indicators
- Personalized practice recommendations

## 🛠 **Technical Features**

### **Data Processing**
- Robust CSV parsing with multiple encoding support
- Automatic column detection and validation
- Error handling and data validation
- Missing data management

### **Visualization Technology**
- **Plotly** for interactive charts and 3D visualizations
- **Streamlit** for responsive web interface
- Real-time chart updates and filtering
- Mobile-friendly responsive design

### **Statistical Analysis**
- Rolling window calculations for trend analysis
- Correlation analysis between performance metrics
- Standard deviation and consistency calculations
- Trend line fitting with slope analysis

### **Smart Scaling**
- Automatic axis configuration based on data range
- Optimized tick marks for readability
- Proper handling of "lower is better" metrics
- Color scales matched to realistic golf performance ranges

## 📈 **Performance Metrics Tracked**

### **Speed Metrics**
- Ball Speed (km/h)
- Club Speed (km/h)
- Smash Factor (efficiency ratio)

### **Distance Metrics**
- Carry Distance (m)
- Distance accuracy and consistency

### **Launch Metrics**
- Launch Angle (°)
- Launch Direction (°)
- Attack Angle (°)

### **Spin Metrics**
- Backspin (rpm)
- Sidespin (rpm)
- Spin Axis (°)

### **Accuracy Metrics**
- Lateral dispersion (m)
- Distance deviation (m)
- Shot shape distribution

## 🎯 **Key Benefits**

### **For Practice**
- **Identify weaknesses** with detailed shot analysis
- **Track improvement** over multiple sessions
- **Focus practice time** on areas needing the most work
- **Visualize progress** with intuitive charts and trends

### **For Performance**
- **Optimize equipment** based on launch conditions
- **Improve consistency** with accuracy tracking
- **Maximize distance** through smash factor optimization
- **Better course management** with realistic performance data

### **For Analysis**
- **Professional-level insights** from practice range data
- **Statistical validation** of improvement efforts
- **Benchmark performance** against optimal ranges
- **Data-driven decision making** for equipment and technique

## 🔧 **Using the Dashboard**

### **Getting Started**
1. **Upload Data**: Use the file uploader to select your Garmin R10 CSV files
2. **Select Club**: Choose a specific club for detailed analysis
3. **Explore Tabs**: Navigate through different analysis views
4. **Interactive Charts**: Hover over data points for detailed information
5. **Track Progress**: Compare sessions to monitor improvement

### **Customizable Settings**
- Rolling window sizes for trend analysis
- Color scales and visualization preferences
- Statistical thresholds for improvement detection
- Chart display options and layouts

### **Smart Defaults**
- Automatic window sizing based on data length
- Optimal axis ranges for golf metrics
- Proper handling of different metric types
- Responsive design for various screen sizes

## 🤝 **Contributing**

We welcome contributions! Please feel free to submit pull requests or open issues for:
- New analysis features
- Visualization improvements
- Bug fixes
- Documentation updates
- Performance optimizations

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- Built with [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for interactive visualizations
- [Pandas](https://pandas.pydata.org/) for data processing
- Designed for [Garmin Approach R10](https://www.garmin.com/en-US/p/689375) launch monitor data

---

## 🏌️‍♂️ **Start Improving Your Golf Game Today!**

Transform your practice sessions into data-driven improvement with comprehensive analytics, beautiful visualizations, and actionable insights. Whether you're a weekend warrior or aspiring professional, this dashboard helps you understand your game like never before.

**Upload your Garmin R10 data and start your journey to better golf!** ⛳

### Kubernetes
```bash
# Deploy
kubectl apply -f k8s-deployment.yaml

# Update to latest image
kubectl set image deployment/garmin-dashboard garmin-dashboard=ghcr.io/rpetersson/garmin-r10-dashboard:latest

# Scale
kubectl scale deployment garmin-dashboard --replicas=3

# Check status
kubectl get pods -l app=garmin-dashboard
```

## GitHub Container Registry

Images are automatically built and published to `ghcr.io/rpetersson/garmin-r10-dashboard` when you push to the main branch.

Available tags:
- `latest` - Latest stable release
- `main` - Latest from main branch
- `v*` - Tagged releases

## Features

- Upload and analyze Garmin R10 CSV data
- Distance and accuracy analytics
- Shot shape classification
- Attack angle optimization
- Interactive charts and statistics

## Architecture

- **Frontend**: Streamlit with Plotly
- **Data**: Pandas with vectorized operations
- **Container**: Multi-stage Docker build
- **CI/CD**: GitHub Actions → GitHub Container Registry
- **Deploy**: Docker, Docker Compose, or Kubernetes

Access the application at `http://localhost:8501` after running.

## Manual Installation

If you prefer to run without Docker:

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/rpetersson/garmin-r10-dashboard.git
cd garmin-r10-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. **Access the Application**: Open your browser to `http://localhost:8501`

2. **Upload Data**: 
   - Use the sidebar to upload one or more CSV files from your Garmin Approach R10
   - Files will be automatically processed and combined

3. **Explore Analytics**:
   - **Summary**: Overview of your golf performance metrics
   - **Distance Analysis**: Club distance charts and statistics
   - **Accuracy Analysis**: Shot dispersion and accuracy metrics
   - **Shot Shape**: Analysis of ball flight patterns (draw, fade, etc.)
   - **Attack Angle**: Iron-specific attack angle analysis
   - **Raw Data**: View and filter your shot data

4. **Optimize Performance**:
   - Review attack angle recommendations for irons
   - Analyze shot shape patterns for consistency
   - Track distance trends across sessions
   - Identify and understand outlier shots

## Data Format

The application expects CSV files from the Garmin Approach R10 with these columns:
- Club Type, Distance, Speed, etc. (standard Garmin R10 output)
- Files are automatically validated and processed

## Key Features Explained

### Shot Shape Classification
- **Straight**: -5° to +5° shot shape
- **Draw/Fade**: -15° to -5° / +5° to +15°
- **Hook/Slice**: Beyond ±15°

### Attack Angle Analysis
- Iron-specific recommendations based on club type
- Visual indicators for optimal attack angles
- Performance correlation analysis

### Performance Optimization
- Vectorized data processing for large datasets
- Smart chart formatting based on data volume
- Automatic outlier detection and filtering

## GitHub Container Registry

This project is automatically built and published to GitHub Container Registry. Images are available at:

- `ghcr.io/rpetersson/garmin-r10-dashboard:latest` - Latest stable release
- `ghcr.io/rpetersson/garmin-r10-dashboard:main` - Latest from main branch
- `ghcr.io/rpetersson/garmin-r10-dashboard:v*` - Tagged releases

### Multi-Platform Support

Images are built for multiple architectures:
- `linux/amd64` (Intel/AMD 64-bit)
- `linux/arm64` (ARM 64-bit, Apple Silicon, Raspberry Pi)

## Development

### Local Development

```bash
# Run in development mode with live reload
make dev

# Or manually
streamlit run app.py --server.fileWatcherType poll
```

### Docker Development

```bash
# Build and test locally
make build
make test

# Development with docker-compose
make compose-dev
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make test`
5. Submit a pull request

## Architecture

- **Frontend**: Streamlit with Plotly charts
- **Data Processing**: Pandas with vectorized operations
- **Containerization**: Multi-stage Docker builds
- **CI/CD**: GitHub Actions with automated testing and publishing
- **Registry**: GitHub Container Registry (ghcr.io)

## Performance Features

- **Vectorized Operations**: 10-100x performance improvement for large datasets
- **Smart Chart Formatting**: Automatic tick spacing based on data volume
- **Efficient Memory Usage**: Optimized pandas operations
- **Caching**: Streamlit caching for improved responsiveness

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the example data files in `/data`

---

**Note**: This application is designed for Garmin Approach R10 CSV data. Other launch monitor formats may require data conversion.
