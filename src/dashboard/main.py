"""Main Streamlit dashboard for FloatChat - ARGO Ocean Data Analysis with Interactive Map and Filters."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from database.models import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="FloatChat - ARGO Ocean Data Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the sophisticated UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .year-tag {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.25rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'selected_years' not in st.session_state:
    st.session_state.selected_years = [1999, 2000, 2001, 2002, 2003]
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = "All"
if 'longitude_range' not in st.session_state:
    st.session_state.longitude_range = [20.09, 144.97]
if 'max_points' not in st.session_state:
    st.session_state.max_points = 1100
if 'argo_data' not in st.session_state:
    st.session_state.argo_data = None

@st.cache_resource
def initialize_database():
    """Initialize database manager."""
    try:
        db_manager = DatabaseManager()
        return db_manager
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return None

def load_argo_data():
    """Load ARGO data based on current filters."""
    if st.session_state.db_manager is None:
        return None
    
    try:
        session = st.session_state.db_manager.get_session()
        
        # Build query based on filters
        query = session.query(st.session_state.db_manager.ARGOProfile)
        
        # Filter by years
        if st.session_state.selected_years:
            query = query.filter(
                st.session_state.db_manager.ARGOProfile.profile_date.in_(
                    [f"{year}-01-01" for year in st.session_state.selected_years]
                )
            )
        
        # Filter by region (simplified for now)
        if st.session_state.selected_region != "All":
            # Add region-specific longitude/latitude filters
            if st.session_state.selected_region == "Arabian Sea":
                query = query.filter(
                    st.session_state.db_manager.ARGOProfile.longitude.between(50, 80),
                    st.session_state.db_manager.ARGOProfile.latitude.between(10, 30)
                )
            elif st.session_state.selected_region == "Bay of Bengal":
                query = query.filter(
                    st.session_state.db_manager.ARGOProfile.longitude.between(80, 100),
                    st.session_state.db_manager.ARGOProfile.latitude.between(5, 25)
                )
            # Add more regions as needed
        
        # Filter by longitude range
        query = query.filter(
            st.session_state.db_manager.ARGOProfile.longitude.between(
                st.session_state.longitude_range[0], 
                st.session_state.longitude_range[1]
            )
        )
        
        # Limit results
        data = query.limit(st.session_state.max_points).all()
        
        # Convert to DataFrame
        if data:
            df_data = []
            for profile in data:
                df_data.append({
                    'platform_number': profile.platform_number,
                    'latitude': profile.latitude,
                    'longitude': profile.longitude,
                    'profile_date': profile.profile_date,
                    'year': profile.profile_date.year if profile.profile_date else None,
                    'region': 'Indian Ocean'  # Simplified for now
                })
            
            df = pd.DataFrame(df_data)
            session.close()
            return df
        else:
            session.close()
            return pd.DataFrame()
                
            except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_map(df):
    """Create interactive map with ARGO data."""
    if df is None or df.empty:
        # Create empty map centered on Indian Ocean
        m = folium.Map(
            location=[15, 75],
            zoom_start=4,
            tiles='OpenStreetMap'
        )
        return m
    
    # Calculate center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
    # Add markers with color coding by year
    for _, row in df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Determine color based on year
            year = row.get('year', 2000)
            if year >= 2020:
                color = 'red'
            elif year >= 2015:
                color = 'orange'
            elif year >= 2010:
                color = 'yellow'
            else:
                color = 'blue'
            
            folium.CircleMarker(
                        [row['latitude'], row['longitude']],
                radius=5,
                popup=f"Platform: {row['platform_number']}<br>"
                      f"Date: {row['profile_date']}<br>"
                      f"Year: {year}",
                tooltip=f"Platform {row['platform_number']}",
                color=color,
                fill=True,
                fillOpacity=0.7
                    ).add_to(m)
            
    return m

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<div class="main-header">🌊 FloatChat - ARGO Ocean Data Analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize database
    if st.session_state.db_manager is None:
        with st.spinner("Initializing database..."):
            st.session_state.db_manager = initialize_database()
    
    if st.session_state.db_manager is None:
        st.error("Failed to initialize database. Please check your configuration.")
        return
    
    # Create two columns: sidebar and main content
    col1, col2 = st.columns([1, 3])
    
            with col1:
        st.markdown("### 🔧 Filters")
        
        # Year Selection
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("📅 **Year Selection**")
        st.markdown("Select years:")
        
        # Year selection with tags
        available_years = list(range(1999, 2024))
        selected_years = st.multiselect(
            "Choose years:",
            options=available_years,
            default=st.session_state.selected_years,
            key="year_selector"
        )
        
        if selected_years != st.session_state.selected_years:
            st.session_state.selected_years = selected_years
                    st.rerun()
        
        # Display selected years as tags
        if st.session_state.selected_years:
            for year in sorted(st.session_state.selected_years):
                st.markdown(f'<span class="year-tag">{year} ✕</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Geographic Region
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("🌍 **Geographic Region**")
        
        region_options = ["All", "Arabian Sea", "Bay of Bengal", "Equatorial Indian Ocean", "Southern Indian Ocean"]
        selected_region = st.selectbox(
            "Select region:",
            options=region_options,
            index=region_options.index(st.session_state.selected_region)
        )
        
        if selected_region != st.session_state.selected_region:
            st.session_state.selected_region = selected_region
            st.rerun()
        
        # Longitude slider
        st.markdown("**Longitude (°E)**")
        longitude_range = st.slider(
            "Longitude range:",
            min_value=20.0,
            max_value=150.0,
            value=st.session_state.longitude_range,
            step=0.1,
            key="longitude_slider"
        )
        
        if longitude_range != st.session_state.longitude_range:
            st.session_state.longitude_range = longitude_range
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Control
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown("📊 **Display Control**")
        
        max_points = st.slider(
            "Max points to display:",
            min_value=100,
            max_value=5000,
            value=st.session_state.max_points,
            step=100,
            key="max_points_slider"
        )
        
        if max_points != st.session_state.max_points:
            st.session_state.max_points = max_points
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Load data based on filters
        if st.session_state.argo_data is None:
            with st.spinner("Loading ARGO data..."):
                st.session_state.argo_data = load_argo_data()
        
        # Data Overview
        st.markdown("### 📊 Data Overview")
        if st.session_state.argo_data is not None and not st.session_state.argo_data.empty:
            df = st.session_state.argo_data
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
                st.metric("Total Profiles", f"{len(df):,}")
        with col2:
                st.metric("Unique Platforms", f"{df['platform_number'].nunique():,}")
        with col3:
                if 'profile_date' in df.columns:
                    date_range = f"{df['profile_date'].min()} to {df['profile_date'].max()}"
                    st.metric("Date Range", date_range)
                else:
                    st.metric("Date Range", "N/A")
        with col4:
                st.metric("Geographic Coverage", "Indian Ocean")
        
        # Data Quality Check
        st.markdown("### 🔍 Data Quality Check")
        if st.session_state.argo_data is not None and not st.session_state.argo_data.empty:
            df = st.session_state.argo_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Ocean Profiles", f"{len(df):,}", "✅")
            with col2:
                indian_ocean_count = len(df[df['region'] == 'Indian Ocean']) if 'region' in df.columns else len(df)
                st.metric("Indian Ocean", f"{indian_ocean_count:,}", "🔵")
            with col3:
                other_count = len(df) - indian_ocean_count
                st.metric("Other Oceans", f"{other_count:,}", "🌊")
        
        # Map Color Legend
        st.markdown("### 🎨 Map Color Legend")
        st.markdown("""
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff0000;"></div>
            <span>Indian Ocean: Blue/Yellow/Orange/Red markers (colored by year: 2020+ = Red, 2015-2019 = Orange, 2010-2014 = Yellow, <2010 = Blue)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #00ff00;"></div>
            <span>Southern Ocean: Green markers</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #800080;"></div>
            <span>Pacific Ocean: Purple markers</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #808080;"></div>
            <span>Other Regions: Gray markers</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Map
        st.markdown("### 🗺️ Interactive Map")
        if st.session_state.argo_data is not None:
            map_obj = create_map(st.session_state.argo_data)
            st_folium(map_obj, width=700, height=500)
        else:
            st.info("No data available. Please adjust your filters.")
    
    # Navigation tabs at the bottom
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Data Visualization", "📊 Analytics", "💬 Chatbot"])
    
    with tab1:
        st.header("📈 Data Visualization")
        if st.session_state.argo_data is not None and not st.session_state.argo_data.empty:
            df = st.session_state.argo_data
            
            # Time series
        if 'profile_date' in df.columns:
                st.subheader("📅 Profiles Over Time")
                time_data = df.groupby(df['profile_date'].dt.to_period('M')).size().reset_index(name='count')
                time_data['period'] = time_data['profile_date'].astype(str)
                
                fig = px.line(time_data, x='period', y='count', title='Number of Profiles by Month')
            st.plotly_chart(fig, use_container_width=True)
        
        # Spatial distribution
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("🌍 Spatial Distribution")
            fig = px.scatter(df, x='longitude', y='latitude', 
                               color='year' if 'year' in df.columns else None,
                               title='ARGO Float Locations',
                               labels={'longitude': 'Longitude (°E)', 'latitude': 'Latitude (°N)'})
                    st.plotly_chart(fig, use_container_width=True)
    else:
            st.info("No data available for visualization. Please adjust your filters.")
    
    with tab2:
        st.header("📊 Analytics")
    if st.session_state.db_manager:
        try:
            session = st.session_state.db_manager.get_session()
            
            # Database statistics
            float_count = session.query(st.session_state.db_manager.ARGOFloat).count()
            profile_count = session.query(st.session_state.db_manager.ARGOProfile).count()
            trajectory_count = session.query(st.session_state.db_manager.ARGOTrajectory).count()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                    st.metric("Total Floats", f"{float_count:,}")
            with col2:
                    st.metric("Total Profiles", f"{profile_count:,}")
            with col3:
                    st.metric("Trajectory Points", f"{trajectory_count:,}")
            
            session.close()
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    with tab3:
        st.header("💬 Chatbot")
        st.info("Chatbot functionality will be available once the RAG system is properly configured.")
        
        # Simple query interface
        query = st.text_input("Ask a question about ARGO data:")
        if st.button("Submit Query"):
            st.info("Query submitted! (RAG system integration pending)")

if __name__ == "__main__":
    main()