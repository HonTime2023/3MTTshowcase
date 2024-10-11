import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set Page Title and Layout
st.set_page_config(page_title="ESAAM App", layout="wide")

# Setting custom settings for UI
st.markdown(
    """
    <style>
    /* Custom Sidebar Style */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF; /* Sidebar background color */
        color: #737B8B; /* Sidebar text color */
    }
    
    /* Custom Background Color */
    .stMain {
        background-color: #C4CECB; /* Background color of app */
    }

    /* Target the header element */
    .st-emotion-cache-h4xjwg {
        background-color: green;  
    }
    
     .st-emotion-cache-yfhhig.ef3psqc5 {
            background-color: green; 
            color: white; 
        }
        .st-emotion-cache-yfhhig.ef3psqc5:hover {
            background-color: darkgreen; 
        }
    
    /* Change h2 color in the sidebar specifically */
   .stSidebar h2 {
       color: #737B8B;  
   }
     /* Change p color in the sidebar specifically */
   .stSidebar p {
       color: #737B8B;  
   }    
    /* Sidebar content */
    .stSidebarContent {
        background-color: #333333;
        color: black;
    }
    /* Logo styling */
    .sidebar-logo {
        width: 42px;
        height; 28px;
        margin: 10px auto;
        display: block;
    }
    /* Flex container for the dashboard */
    .dashboard-container {
        display: flex;
        border-radius: 10px;
        justify-content: center;
        align-items: center;
        background-color: green;
        color: white;
        padding: 10px;
        margin-bottom: 20px;
    }
     /* Style for the file uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFF0EE;  
        border: 2px #007BFF;  
        border-radius: 10px;         
        padding: 20px;               
        text-align: center;          
           }
           
    [data-testid="stFileUploaderDropzone"] .st-emotion-cache-9ycgxx {
        color: black;  /* Change to black */
        text-align: left;
    }
    /* Change color of the small text (instructions) */
    [data-testid="stFileUploaderDropzone"] small {
        color: black;  /* Change to your desired text color */
    }
    
    /* Style the button within the dropzone */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: green;  /* Button background color */
        color: white;                /* Button text color */
        border: none;                /* No border */
        border-radius: 5px;         /* Rounded edges */
        padding: 10px 15px;         /* Button padding */
        cursor: pointer;             /* Pointer cursor on hover */
    }
    
    /* Hover effect for the button */
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #AFCF35;
        color: black;  
    } 
    
    /* Banner containers*/
         
        .banner-main{
            display: flex;
            justify-content: space-between;
           background: linear-gradient(90deg, #29592C, #27A258, #29592C);
             border-radius: 10px;
             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 100px;
            padding: 20px;
            position: relative;
             transition: 0.5s ease;
             width: 100%;
             
            
            }
        .banner-title {
             margin: 0;
            color: var(--bs-primary);
            font-family: var(--font-family-title);
            font-weight: bold;
            font-size: 30px;
        }
        .banner-item-header p {
            margin-top: -5px;
            color: var(--bs-gray);
            /*margin-left: 30px;*/
            /*font-size: 1rem;*/
        }
         .banner-image{
          position: absolute;
          bottom: 0;
         }
        .banner-image img {
            width: 32%;
            float: right;
            transition: 0.5s ease;
            margin-right: 10%;
            
        }
         .banner-main:hover img {
         filter: grayscale(100%);
         transition: 0.5s ease;
        }
        .tab, .book {
            height: 40px;
        }
        .tab{
            position: absolute;
            right: 0;
            top: 30%;
            }
        .book{
        position: absolute;
        top: 0;
        left: 50%;
        transform: translate(-50%, -50%)}
         
        /* General Styles for Flexbox */
        .dlab-media {
            display: flex;
            background-color: #C4CECB;
            border-radius: 10px;
            flex-wrap: wrap;
            gap: 5px;
        }
        .banner-item {
            flex-grow: 1;
        }
        .banner-item-header {
        }
       
     
        .sub-banner {
            background-red: #F75A5B;
            color: white;
            font-size: 12px;
            font-size: 20px;
            margin-top: 30px;
            padding: 20px;
            }  
        .sub-banner-contents {
            
                       
            } 
         
        .metric-box {
        background-color: #f0f8ff; /* Light blue background */
        border-radius: 10px; /* Rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Box shadow */
        padding: 20px; /* Internal padding */
        margin: 10px; /* External margin */
        text-align: center; /* Center the text */
            }
        .metric-label {
        font-size: 18px; /* Font size for the label */
        color: black; 
        }
        .metric-value {
        background-color: #f0f8ff;
        font-size: 32px; 
        font-weight: bold; 
        color: black; 
            }
             .eda-heading {
        display: flex;
        align-items: center;
        color: black; /* Black text for the heading */
        }
        .eda-heading .icon {
            font-size: 25px; /* Adjust icon size to match the heading */
            margin-right: 10px; /* Add some spacing between the icon and text */
            color: black; /* Set the icon color to black */
        }
        .eda-heading h2 {
            color: black;
            margin: 0; /* Ensure no margin is added to the heading */
        }
        .stMetric { 
            background-color: #f0f8ff; /* Light blue background */
            color: black;
            border-radius: 10px; /* Rounded corners */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Box shadow */
            padding: 20px; /* Internal padding */
            margin: 5px; /* External margin */
            text-align: left
            font-size: 10 px;
        }
        
        .st-emotion-cache-efbu8t {  /* Metric value class */
            font-size: 15px; 
            font-weight: bold; 
            color: black; 
        }  
       .stMarkdownContainer {
            font-size: 18px; /* Font size for the label */
            color: black; 
            }
        .st-emotion-cache-1wivap2 { /* Delta value class */
            font-size: 16px; /* Smaller font size for delta */
            margin-top: 5px; /* Space between value and delta */
        }
        /* Adjust the size of the delta arrow */
        .stMetric svg {
            width: 10px;  /* Adjust the width as needed */
            height: 10px; /* Adjust the height as needed */
        }
        .st-emotion-cache-uef7qa p {
            color: #777B8B !important;
            font-size: 10px !important;
            font-weight: bold !important;
            }
        .st-emotion-cache-1inwz65  {
            color: black;
            }
        
    </style>
    """, unsafe_allow_html=True
)


# Banner layout using the flexbox design
st.markdown(
    """
    <div class="banner-container">
        <div class="banner-item">
            <div class="dlab-media d-flex justify-content-between">
                <!-- Content Section -->
                <!-- gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg-->
                <div class="banner-main">
                    <div class="banner-item-header">
                        <p class="banner-title">Welcome to ESAAM Soft</p>
                        <p>Exploratory and Sensitivity Analysis App for 3MTT</p>
                    </div>    
                    <!-- Image Section -->
                 <div class="banner-image">
                        <img src="https://app.3mtt.training/static/media/education-girl.b8a777ab708c361de94c.png", alt="banner-image" class=" ">
                    </div> 
                 </div>
                <!-- Icons Section -->
                 <div class="banner-icons">
                    <img class="book" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAqCAYAAADvczj0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAI1SURBVHgB7Zq7jsIwEEUnqxQpKCgoUlBQ7Efs/1f7CRRbUFBQUKSgSBGJnWtmliGEV2J7JeMjjYJ42HPtiR1xXRBzPB7nfEFUHKUEXg/RmquNQ1EUHXmGc9NcZnKtTH4lPZkj57bDGwU3+MnXJfnBCedoENzJgV6E84GQBZ0E6iT4YMf5rDFCvsQCnQEkjOQxAI101tz6kVTYQsKXwD41x7qksFTSUS3i9xxbFt9KqS7l81Airwgt2AJRELhksRA+j9y/I3qHwoL+iQ96M7Lg1MmCUycLTp0sOHWy4NTJglMnC06dLDh1suDUyYJTB4Ibeg+c/VKItbGik7WBv21hj3R06c8A/bPc+jr6m5hYP4vMVbHemL6GrwQDoCtoIuIgzOjsBanp5QNU30ECXlVLE5kseIgJXhEqay/RhHAjgwi2iPha4haYyQ0FslwtV4KlRO/5w+7+fjUxs1bUph01116yVaUtGpOjLlowucZYlbrAqSfcPkreDGh3b9B6a4PGvYm4hVq2G6wBEPxFfu1KHQC9D5+ePXMSQcMnmIxvHW2foL2/hI0pPli6ZoGrKewWV6FqMMMo59VAZ/19ruzFGNDWhk4VoKv4jMbTSViG8sN3fnDOwy1a5n5xDTyz38m9bw+b6D7sG+Tk1gc678kPczSHYchWltdtyQwcynPsgRS7Bux9PGxYgu7DZg9G6T66DaLsxcEfPICUP8Sv6HLWIWxL8pxLEYgi2MLi9dSOO9UTS6jyC3f33xkG4DF9AAAAAElFTkSuQmCC", alt="icon" class=" ">
                    <img class="tab" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAA6CAYAAADvEjRHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAK4SURBVHgB7Zsvc+MwEMWfOwEBAQUFBwIMDgQcONDvjwsKAgoKCgwCDAoMCgIyk+7byBnX0R/H6nQae9+MxhPZm7V/llZSsiqQoePxuJLDXyn3UvZS3oqieB9oW8phLWUhpXa2hwF2fZ+V2NXIUIGRkptZyuERp4foais31SRsSzmUvepG7LZI+/wvZXmtz5juMF4PuARA/UFaa0/dvXvIlM/lSJ9B5UBYYLxCtikIY+2iyoEwGUVjgjRPBh+WlStdLeB/MwdXYgq9uZRtyCe193z+kPKeihdeCO7hS5wATEHRUeQCggDg8LPGNOUdir9AEAAbZEbaGxBbxXMXxDkwurF76gAoxqN/3QqFIAD48CXmI85Jzl2+jbRlwojRldPhbl96cMV3bWoauwnUV7iM8hjgk6PArlfHoB5r2aWAqNktFm4kCA1ZvKFX3xDjZne+G9qn5vIu9vjUxIaziM+Dx2ct11fwT7MpNgC2hordIUYra04+QskF1DWSe+dL3Ea+V6cAhLAKXFC7LwmpubI+dQ1b0McIOyrY8twz7AKnkxCiS2LXQqpedT1wWfuKr32fb+olZZThM3ieXSy2CEo2TXFeMbjgBHLIm2ztCODJxSPeQzPkt4SxPulPbILnc1aCZweIR/SY7ah4k+PTJ1tFIt4SlgN+5JiEYhA2mImsO8AgqAwCDILKIMAgqAwCDILKIMAgqAwCDILKIMAgqAwCDILKIMAgqAwCDILKIMAgqAwCDILKIMAgqAjhW7NDblGzh8C/+QnhJ3OSfps0wYMQsnaN3Lg0JenOZYt8W9bHjUkbQDs6VJifqjY7TyG47K85dQsmfFXth+484Q0uUExcbYLnWf1Uf6bvcL/DVLPd+ZJf+kmqoZ0vbdb7VBK3OBfadbtAV6k9UG1WuW8P1G9XuwdKs+5jyaKf2cDeq9bZQeoAAAAASUVORK5CYII=", alt="icon" class=" ">
                </div> 
            </div>
        </div>
        
    </div>
    
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown('<img class="sidebar-logo" src="https://app.3mtt.training/static/media/main.242b8b1ce339b38fd589.png" alt="Logo">', unsafe_allow_html=True)
# Include Font Awesome for icons
st.sidebar.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div class="dashboard-container" style="display: flex; align-items: center;">
        <i class="fas fa-tachometer-alt" style="font-size: 24px; margin-right: 8px;"></i>
        <span>Dashboard</span>
    </div>
""", unsafe_allow_html=True)

# Sidebar for Upload and Navigation Options
st.sidebar.header("Upload Data and Select Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Sidebar Options for EDA and Scenario Analysis
analysis_type = st.sidebar.selectbox("Choose Analysis Type", ('EDA Dashboard', 'Sensitivity Analysis'))

# Function to format large numbers into K, M, B
def format_number(num):
    if abs(num) >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.1f}Tn"
    elif abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}Bn"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}Mn"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

# Load and Clean Data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.dropna()  # Remove missing rows
    data = data.reset_index(drop=True)
    data.index = data.index + 1
    data.index.name = 'S/N'
    st.sidebar.success('Data Uploaded and Cleaned Successfully...!')

    # Predefined course-to-salary mapping
    course_mapping = {
        'Project Manager': 'Product Management',
        'Sustainable Tech': 'Cloud Computing',
        'Software Developer': 'Software Development',
        'Cyber Security': 'Cybersecurity',
        'UI/UX': 'UI/UX Design',
        'Entrepreneurship': 'Product Management',
        'Administrative professional': 'Quality Assurance',
        'Generative AI': 'AI / Machine Learning',
        'Soft skills': 'Product Management',
        'Data analyst': 'Data Analysis & Visualization',
        'Career Essentials in Entrepreneurship': 'Product Management',
        'Systems administrator': 'DevOps',
        'AI/ML': 'AI / Machine Learning',
        'Business analyst': 'Data Analysis & Visualization',
        'Animation': 'Animation',
        'Cloud Computing': 'Cloud Computing',
        'UI/UX Design': 'UI/UX Design',
        'Data Analysis & Visualization': 'Data Analysis & Visualization',
        'Data Science': 'Data Science',
        'DevOps': 'DevOps',
        'Game Development': 'Game Development',
        'Product Management': 'Product Management',
        'Quality Assurance': 'Quality Assurance',
        'Software Development': 'Software Development',
        'Cybersecurity': 'Cybersecurity'

    }

    salary_mapping = {
        'AI / Machine Learning': 185000,
        'Animation': 102083,
        'Cloud Computing': 207000,
        'UI/UX Design': 119000,
        'Data Analysis & Visualization': 116666,
        'Data Science': 112740,
        'DevOps': 306000,
        'Game Development': 150000,
        'Product Management': 200000,
        'Quality Assurance': 162000,
        'Software Development': 150000,
        'Cybersecurity': 198000
    }

    # MPI data and other parameters
    pi_data = {
        'State of Residence': ['Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
                       'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa',
                       'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger',
                       'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe',
                       'Zamfara', 'Federal Capital Territory'],

        'MPI': [0.101, 0.283, 0.293, 0.109, 0.298, 0.401, 0.312, 0.315, 0.299, 0.173, 0.320, 0.126, 0.125, 0.234, 0.380,
        0.142, 0.385, 0.298, 0.270, 0.304, 0.385, 0.250, 0.185, 0.101, 0.243, 0.278, 0.289, 0.095, 0.150, 0.190,
        0.365, 0.241, 0.409, 0.340, 0.370, 0.328, 0.186],

        'Average Household Size': [4.8, 6.1, 4.5, 4.4, 6.2, 4.9, 5.7, 6.2, 4.7, 4.5, 4.4, 4.5, 4.3, 4.4, 6.1, 4.4, 6.9,
                                   5.7, 5.7, 6.3, 6.6, 4.7, 4.5, 4.2, 5.3, 6.5, 4.4, 4.5, 4.3, 4.4, 4.8, 4.3, 6.4, 6.5,
                                   6.6, 6.5, 4.6],
        'Dependency Ratio': [0.94, 1.14, 0.89, 0.86, 1.16, 0.95, 1.08, 1.15, 0.91, 0.88, 0.85, 0.88, 0.84, 0.86, 1.12,
                             0.85, 1.23, 1.04, 1.04, 1.14, 1.25, 0.91, 0.87, 0.82, 1.03, 1.19, 0.85, 0.88, 0.84, 0.87,
                             0.98, 0.84, 1.17, 1.18, 1.23, 1.22, 0.82]
    }

    # Prepare the Data
    required_columns = ['Name', 'Course', 'Fellow ID', 'Cohort', 'State of Residence', 'Gender']
    if all(col in data.columns for col in required_columns):
        data = data[required_columns]
        data['Mapped Course'] = data['Course'].map(course_mapping)
        data['Salary'] = data['Mapped Course'].map(salary_mapping)

        pi_df = pd.DataFrame(pi_data)
        data = data.merge(pi_df, on='State of Residence', how='left')

        # Generate completion status using Bernoulli distribution
        data['Completion Status'] = bernoulli.rvs(0.73, size=len(data))

        # Determine employment status based on completion status
        # First, create a temporary column for potential employment candidates
        employment_candidates = data[data['Completion Status'] == 1]

        # Apply Bernoulli distribution for employment status only to completed fellows
        employment_candidates['Employment Status'] = bernoulli.rvs(0.11, size=len(employment_candidates))

        # Merge back the employment status into the original data
        data = data.merge(employment_candidates[['Fellow ID', 'Employment Status']], on='Fellow ID', how='left')

        # Fill NaN values in Employment Status for non-completed fellows with 0
        data['Employment Status'] = data['Employment Status'].fillna(0)

        # Calculate SP Ratio only for employed fellows (Employment Status == 1)
        data['SP Ratio'] = np.where(data['Employment Status'] == 1, data['Salary'] / 53078.84, 0)

    else:
        # Custom error design using markdown and CSS
        st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #FFDDDD; border-left: 5px solid red; font-family: Arial;">
                <span style="font-size: 16px; color: red;">⚠️ Error: </span><span style="font-size: 14px; color: black;">Please Upload the Specified 3MTT Dataset with Name, Course, Fellow ID, Cohort, State of Residence, and Gender</span>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    # If EDA Dashboard is selected
    if analysis_type == 'EDA Dashboard':

        # Function to create a custom metric display
        def custom_metric(label, value):
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


        # EDA Contents
        total_fellows = data.shape[0]
        total_states = data['State of Residence'].nunique()
        total_females = data[data['Gender'] == 'Female'].shape[0]
        total_males = data[data['Gender'] == 'Male'].shape[0]


        # Header for the EDA section
        st.markdown('''
        <div class="eda-heading">
            <i class="fas fa-bars icon"></i>
            <h2>Exploratory Data Analysis</h2>
        </div>
        ''', unsafe_allow_html=True)

        # Display the Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            custom_metric("Total Fellows", format_number(total_fellows))
        with col2:
            custom_metric("Total States", total_states)
        with col3:
            custom_metric("Total Females", format_number(total_females))
        with col4:
                custom_metric("Total Males", format_number(total_males))
        with col5:
            custom_metric("Average MPI", 0.257)

        # Doughnut chart showing percentage of males vs females
        gender_counts = data['Gender'].value_counts()
        fig1 = px.pie(gender_counts, values=gender_counts, names=gender_counts.index, hole=0.4,
                      title="Percentage of Males vs Females")
        # Update the layout to change the background color
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Background color outside the chart
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background inside the chart
            title_font=dict(size=20, color='black'),  # Title font color set to black
            legend=dict(
                font=dict(color='black'),  # Legend text color set to black
                    )
        )
        # Set custom colors for slices if needed
        fig1.update_traces(
            marker=dict(
                colors=['#008000', '#AFCF35'],  # Custom colors for slices
            )
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1)

        # Horizontal bar chart showing number of males and females per cohort
        cohort_gender = data.groupby(['Cohort', 'Gender']).size().unstack().fillna(0)
        fig2 = px.bar(cohort_gender,
                      orientation='h',
                      title="Males and Females per Cohort",
                      labels={'value': 'Number of Fellows'},
                      color_discrete_map={'Male': 'green', 'Female': '#AFCF35'})  # Custom bar colors for Male and Female

        # Updating layout for text and axis customizations
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Background color outside the chart
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background inside the chart
            title=dict(
                text="Males and Females per Cohort",
                font=dict(size=20, color='black')),  # Title color
            legend=dict(
                title_text='Gender',
                font=dict(color='black')  # Legend text color
            ),
            xaxis=dict(
                title='Number of Fellows',
                titlefont=dict(color='black'),
                tickfont=dict(color='black')  # X-axis labels color
            ),
            yaxis=dict(
                title='Cohort',
                titlefont=dict(color='black'),
                tickfont=dict(color='black')  # Y-axis labels color
            )
        )

        col2.plotly_chart(fig2)

        # Fellows by Course
        course_counts = data['Mapped Course'].value_counts()
        course_counts_df = pd.DataFrame(course_counts).reset_index()
        course_counts_df.columns = ['Course', 'Count']

        # Create a bar chart with Plotly
        fig = px.bar(course_counts_df, x='Course', y='Count', title="Fellows by Course",
                     labels={'Course': 'Course', 'Count': 'Number of Fellows'})

        # Customize the plot: transparent background, axes labels in black, axes in black, and bars in green
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            xaxis=dict(
                title='Course',
                titlefont=dict(color='black'),  # X-axis label color
                tickfont=dict(color='black'),  # X-axis tick labels color
                linecolor='black'  # X-axis line color
            ),
            yaxis=dict(
                title='Number of Fellows',
                titlefont=dict(color='black'),  # Y-axis label color
                tickfont=dict(color='black'),  # Y-axis tick labels color
                linecolor='black'  # Y-axis line color
            ),
            title=dict(
                font=dict(size=20, color='black')  # Title text color
            )
        )

        # Update bar color to green
        fig.update_traces(marker_color='green')

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Gauge showing percentage of total fellows out of 3 million (3MTT_Target)
        total_fellows = len(data)
        max_cap = 3000000
        percentage_fellows = (total_fellows / max_cap) * 100
        # Create the gauge chart with custom styling
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage_fellows,
            title={'text': "Total Fellows as % of 3 Million", 'font': {'color': 'black'}},  # Title in black
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'black', 'tickfont': {'color': 'black'}},
                # Axis labels in black
                'bar': {'color': 'green'},  # Gauge bar color (fill)
                'bordercolor': 'green',  # Outline color of the gauge
                'bgcolor': '#AFCF35',  # Background color inside the gauge
                'borderwidth': 2,  # Width of the green outline
                'steps': [{'range': [0, 100], 'color': 'rgba(0, 255, 0, 0.1)'}]
                # Simulate shadow by using a semi-transparent green
            },
            number={'font': {'color': 'black'}}  # Value color in black
        ))

        # Customize layout for background
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Green background
            margin=dict(l=30, r=30, t=50, b=50),  # Space around the chart to simulate a shadow
            font=dict(color='black'),  # Set font for labels in black
            plot_bgcolor='white',  # White plot background for contrast inside the gauge
            # Adding a shadow effect by using a surrounding background contrast
            xaxis_showgrid=False, yaxis_showgrid=False  # Hide grids to clean up the visual
        )
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig3)

        # Calculate the average MPI for each state, grouped by gender
        state_mpi = data.groupby(['State of Residence', 'Gender'])['MPI'].mean().reset_index()

        # Calculate the overall average MPI per state (ignoring gender for sorting)
        state_avg_mpi = state_mpi.groupby('State of Residence')['MPI'].mean().sort_values(ascending=False)

        # Get the top 5 states with the highest average MPI
        top5_states = state_avg_mpi.head(10).index

        # Filter the data to include only the top 5 states
        top5_states_mpi = state_mpi[state_mpi['State of Residence'].isin(top5_states)]

        # Ensure the states are displayed in the correct order by setting the category order
        fig4 = px.bar(
            top5_states_mpi,
            x='MPI',
            y='State of Residence',
            color='Gender',
            title='Top 5 States by Average MPI and Gender',
            labels={'MPI': 'Average MPI', 'State of Residence': 'State'},
            orientation='h',  # Horizontal bar chart
            barmode='stack',  # Stack the bars by gender
            color_discrete_map={'Male': 'green', 'Female': '#AFCF35'}
        )

        # Update the layout to display states in descending order of their average MPI
        fig4.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': top5_states})

        # Customize the plot: transparent background, axes labels in black, axes in black, and bars in green
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            legend=dict(
                title_text='Gender',
                font=dict(color='black')  # Legend text color
            ),
            xaxis=dict(
                title='Average MPI',
                titlefont=dict(color='black'),  # X-axis label color
                tickfont=dict(color='black'),  # X-axis tick labels color
                linecolor='black'  # X-axis line color
            ),
            yaxis=dict(
                title='State',
                titlefont=dict(color='black'),  # Y-axis label color
                tickfont=dict(color='black'),  # Y-axis tick labels color
                linecolor='black'  # Y-axis line color
            ),
            title=dict(
                font=dict(size=20, color='black')  # Title text color
            )
        )

        # Display the chart
        col2.plotly_chart(fig4)

        # Bar chart by State and Gender
        fig2 = px.bar(
            data,
            x='State of Residence',
            color='Gender',
            title="Fellows by State",
            color_discrete_map={'Male': 'green', 'Female': '#AFCF35'}  # Customize colors for each gender
        )

        # Customize the plot: transparent background, axes labels in black, and title in black
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            legend=dict(
                title_text='Gender',
                font=dict(color='black')  # Legend text color
            ),
            xaxis=dict(
                title='State of Residence',
                titlefont=dict(color='black'),  # X-axis label color
                tickfont=dict(color='black'),  # X-axis tick labels color
                linecolor='black'  # X-axis line color
            ),
            yaxis=dict(
                title='Number of Fellows',
                titlefont=dict(color='black'),  # Y-axis label color
                tickfont=dict(color='black'),  # Y-axis tick labels color
                linecolor='black'  # Y-axis line color
            ),
            title=dict(
                font=dict(size=20, color='black')  # Title text color
            )
        )

        # Display the bar chart in Streamlit
        st.plotly_chart(fig2)

        # Line chart showing total number of fellows for each course across the states
        course_state_totals = data.groupby(['State of Residence', 'Mapped Course']).size().unstack(fill_value=0)
        fig5 = px.line(course_state_totals, title="Total Number of Fellows for Each Course Across States",
                       labels={'value': 'Number of Fellows'})
        # Customize the line chart: transparent background, axes labels in black, and title in black
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            legend=dict(
                title_text='Gender',
                font=dict(color='black')  # Legend text color
            ),
            xaxis=dict(
                title='Mapped Course',
                titlefont=dict(color='black'),  # X-axis label color
                tickfont=dict(color='black'),  # X-axis tick labels color
                linecolor='black'  # X-axis line color
            ),
            yaxis=dict(
                title='Number of Fellows',
                titlefont=dict(color='black'),  # Y-axis label color
                tickfont=dict(color='black'),  # Y-axis tick labels color
                linecolor='black'  # Y-axis line color
            ),
            title=dict(
                font=dict(size=20, color='black')  # Title text color
            )
        )

        st.plotly_chart(fig5)


        # Add an interactive filterable table
        st.markdown("<h5 style='color: black; margin-bottom: 0.5px;'>Cleaned Data Table (Filterable)</h5>", unsafe_allow_html=True)
        filtered_data = st.dataframe(data)  # Display the table

        # Filter options
        st.markdown("<h5 style='color: black; margin-bottom: 0.5px;'>Filter Options</h5>",
                    unsafe_allow_html=True)
        state_filter = st.selectbox("Select State of Residence",
                                    options=['All States'] + list(data['State of Residence'].unique()),
                                    label_visibility="collapsed")
        gender_filter = st.selectbox("Select Gender", options=['All Gender'] + list(data['Gender'].unique()),
                                     label_visibility="collapsed")
        cohort_filter = st.selectbox("Select Cohort", options=['All Cohorts'] + list(data['Cohort'].unique()),
                                     label_visibility="collapsed")
        course_filter = st.selectbox("Select Course", options=['All Courses'] + list(data['Mapped Course'].unique()),
                                     label_visibility="collapsed")


        # Add a button to apply the filter
        if st.button("Filter"):
            # Apply filters
            filtered_data = data.copy()

            if state_filter != 'All':
                filtered_data = filtered_data[filtered_data['State of Residence'] == state_filter]
            if gender_filter != 'All':
                filtered_data = filtered_data[filtered_data['Gender'] == gender_filter]
            if cohort_filter != 'All':
                filtered_data = filtered_data[filtered_data['Cohort'] == cohort_filter]
            if course_filter != 'All':
                filtered_data = filtered_data[filtered_data['Course'] == course_filter]

            # Reset index and add an 'S/N' column starting from 1
            filtered_data.reset_index(drop=True, inplace=True)
            filtered_data.index += 1
            filtered_data.index.name = 'S/N'

            # Display filtered data
            st.subheader("Filtered Data Table (Filterable)")
            st.dataframe(filtered_data)


            # Add a download button for the filtered data
            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')


            csv = convert_df(filtered_data)

            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name='filtered_data.csv',
                mime='text/csv',
            )
    # If Sensitivity Analysis is selected
    elif analysis_type == 'Sensitivity Analysis':
        st.markdown('<h2 style="color: black;"><i class="fas fa-chart-line"></i> Sensitivity Analysis</h2>', unsafe_allow_html=True)

        # Combine features as described (in documentation)
        data['Education_Combined'] = data['Completion Status']/0.2
        data['LivingStandards_Combined'] = data['SP Ratio']/0.198
        data['FoodSecurity_Combined'] = data['SP Ratio'] /0.2
        data['RuralLivelihoods_Combined'] = (data['SP Ratio']-data['Average Household Size']-data[
            'Dependency Ratio'])/0.04
        data['Risk_Combined'] = data['SP Ratio'] /0.05

        # Prepare the dataset for LightGBM
        y = data['MPI']
        X_combined = data[['Education_Combined', 'LivingStandards_Combined', 'FoodSecurity_Combined',
                           'RuralLivelihoods_Combined', 'Risk_Combined']]
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

        # LightGBM Model Training
        try:
            model = lgb.LGBMRegressor()
            model.fit(X_train, y_train)
            st.markdown(
                '<div style="color: white; background-color: #4CAF50; padding: 10px; border-radius: 5px;">'
                'Model training completed successfully..!'
                '</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error in training LightGBM model: {e}")

        # Make predictions
        try:
            y_pred = model.predict(X_test)

            # Evaluate the model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Feature Importance and Interpretability with SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except Exception as e:
            st.error(f"Error in SHAP analysis: {e}")


        # Function to create a custom metric display
        def custom_metric(label, value):
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # Create three columns for Accuracy Analysis
        col_acc1, col_acc2, col_acc3 = st.columns(3)

        # Display metrics in three columns
        with col_acc1:
            custom_metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")

        with col_acc2:
            custom_metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")

        with col_acc3:
            custom_metric(label=" R-squared Value [R²]  (Accuracy)", value=f"{r2:.4f}")

        # Plot Actual vs Predicted MPI
        # Create a DataFrame for actual and predicted MPI
        results_df = pd.DataFrame({'Actual MPI': y_test, 'Predicted MPI': y_pred})

        # Ensure index alignment with the original data's states
        results_df['State of Residence'] = data['State of Residence'].iloc[y_test.index].values

        # Group by state and calculate the mean actual and predicted MPI
        results_df = results_df.groupby('State of Residence')[
                    ['Actual MPI', 'Predicted MPI']].mean().reset_index()

        # Create the line chart with Plotly
        fig10 = go.Figure()

        # Add the actual MPI line
        fig10.add_trace(go.Scatter(x=results_df['State of Residence'],
                                 y=results_df['Actual MPI'],
                                 mode='lines',
                                 name='Actual MPI',
                                 line=dict(color='#AFCF35'),
                                 ))

        # Add the predicted MPI line
        fig10.add_trace(go.Scatter(x=results_df['State of Residence'],
                                 y=results_df['Predicted MPI'],
                                 mode='lines',
                                 name='Predicted MPI',
                                 line=dict(color='green'),
                                 ))

        # Update layout to customize the chart
        fig10.update_layout(
            title='Actual vs Predicted MPI per State',
            title_font_color='black',
            title_font_size=20,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            xaxis_title='State of Residence',
            yaxis_title='MPI',
            xaxis=dict(title_font_color='black', tickfont=dict(color='black')),
            yaxis=dict(title_font_color='black', tickfont=dict(color='black')),
            legend=dict(
                title_text='MPI',
                font=dict(color='black')  # Legend text color
            ),

        )

        # Display the line chart in Streamlit
        st.plotly_chart(fig10)


        # Display Feature Importance using SHAP
        shap_importance = pd.DataFrame(
            {'Feature': X_combined.columns, 'Importance': np.abs(shap_values).mean(axis=0)}
        )
        shap_importance['Normalized_Weight'] = shap_importance['Importance'] / shap_importance['Importance'].sum()
        # Create the bar chart using Plotly
        fig = go.Figure()

        # Add bars for SHAP feature importance
        fig.add_trace(go.Bar(
            x=shap_importance['Feature'],  # Feature names on X-axis
            y=shap_importance['Normalized_Weight'],  # Normalized weights on Y-axis
            marker_color='green',  # Set bar color to green
        ))

        # Customizing the layout
        fig.update_layout(
            title='Feature Importance using SHAP',
            title_font=dict(color='black', size=20),
            xaxis_title='Features',
            yaxis_title='Normalized Importance',
            xaxis=dict(showline=True, linecolor='black', tickfont=dict(color='black')),
            yaxis=dict(showline=True, linecolor='black', tickfont=dict(color='black')),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
            legend=dict(font=dict(color='black')),  # Legend text color (though no legend for this bar chart)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Create Cards for Sensitivity Analysis
        previous_tot_fellows = data.shape[0]  # From uploaded data
        previous_total_persons_impacted = 10000  # Dummy initial value
        previous_total_salary = 5000000  # Dummy initial value
        previous_avg_actual_mpi = 0.257  # Nigeria's Average MPI
        previous_avg_predicted_mpi = 0.251 # Educated Dummy using MAE Value
        previous_avg_mpir = 0.05  # Dummy initial value

        # Sliders for user input
        # Header for Sensitivity Analysis

        # Load Font Awesome for icons
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
            """, unsafe_allow_html=True)

        # Inline markdown for the header with icon and text
        st.markdown('''
            <div style="background-color: green; width: 100%; color: white; padding: 10px; border-radius: 10px; display: inline-flex; align-items: center;">
                <i class="fas fa-sliders-h" style="font-size: 15px; margin-right: 10px;"></i>
                <h4 style="margin: 0; color: white;">Adjust Desired Variables</h4>
            </div>
        ''', unsafe_allow_html=True)
        st.write("  ")
        st.write(" ")
        # Sliders displayed together below the comparison line chart
        with st.container():
            col_slider1, col_slider2 = st.columns(2)
            with col_slider1:
                num_fellows = st.slider("Number of Fellows", min_value=1, max_value=33000000, value=100)
                sp_ratio = st.slider("SP Ratio", min_value=0.0, max_value=10.0, value=0.5)
            with col_slider2:
                completion_rate = st.slider("Completion Rate (%)", min_value=0, max_value=100, value=73) / 100  # Set to default from cohort one
                employment_rate = st.slider("Employment Rate (%)", min_value=0, max_value=100, value=11) / 100  # Set to default from cohort one


        # Function to generate Bernoulli distributed values based on slider input
        def generate_bernoulli(probability, size):
            return np.random.binomial(1, probability, size=size)


        # Step 1: Get the distribution of each column from the original dataset
        state_distribution = data['State of Residence'].value_counts(normalize=True).to_dict()
        gender_distribution = data['Gender'].value_counts(normalize=True).to_dict()
        mapped_course_distribution = data['Mapped Course'].value_counts(normalize=True).to_dict()
        salary_distribution = data['Salary'].value_counts(normalize=True).to_dict()
        household_size_distribution = data['Average Household Size'].value_counts(normalize=True).to_dict()
        dependency_ratio_distribution = data['Dependency Ratio'].value_counts(normalize=True).to_dict()


        # Step 2: Function to sample based on distribution
        def sample_from_distribution(distribution, size):
            return np.random.choice(list(distribution.keys()), size=size, p=list(distribution.values()))


        # Step 3: Create a new DataFrame for adjusted fellows
        data_adjusted = pd.DataFrame()

        tot_fellows = 100  # Simulations will be across a sample 100

        # Step 1: Calculate the distribution of 'State of Residence' in the original dataset
        state_distribution_series = data['State of Residence'].value_counts(normalize=True)

        # Step 2: Calculate the number of samples needed from each state based on the distribution
        samples_per_state = (state_distribution_series * tot_fellows).round().astype(int)

        # Step 4: Stratified sampling for each state
        for state, count in samples_per_state.items():
            # Sample 'count' entries from the original data for the specific state
            state_samples = data[data['State of Residence'] == state].sample(n=count, replace=True, random_state=42)
            # Append these samples to the new DataFrame
            data_adjusted = pd.concat([data_adjusted, state_samples], ignore_index=True)

        # If the total number of samples is less than tot_fellows due to rounding, add random samples
        if len(data_adjusted) < tot_fellows:
            additional_samples = data.sample(n=tot_fellows - len(data_adjusted), replace=True, random_state=42)
            data_adjusted = pd.concat([data_adjusted, additional_samples], ignore_index=True)

        # Step 5: Randomly assign other features based on distributions
        data_adjusted['Gender'] = np.random.choice(
            list(gender_distribution.keys()),
            size=len(data_adjusted),
            p=list(gender_distribution.values())
        )
        data_adjusted['Mapped Course'] = np.random.choice(
            list(mapped_course_distribution.keys()),
            size=len(data_adjusted),
            p=list(mapped_course_distribution.values())
        )
        # Generate 'Completion Status' based on 'completion_rate' slider
        data_adjusted['Completion Status'] = generate_bernoulli(completion_rate, size=len(data_adjusted))

        # Generate 'Employment Status' based on 'employment_rate' slider
        data_adjusted['Employment Status'] = generate_bernoulli(employment_rate, size=len(data_adjusted))

        # Assign 'Salary' to employed individuals based on 'Mapped Course'
        # First, calculate average salary by 'Mapped Course' from the original data for employed individuals
        salary_by_course = data[data['Employment Status'] == 1].groupby('Mapped Course')['Salary'].mean()
        salary_by_course_dict = salary_by_course.to_dict()

        # Initialize 'Salary' to zero
        data_adjusted['Salary'] = 0

        # Assign 'Salary' to employed individuals
        employed_indices = data_adjusted['Employment Status'] == 1
        data_adjusted.loc[employed_indices, 'Salary'] = data_adjusted.loc[employed_indices, 'Mapped Course'].map(
            salary_by_course_dict).fillna(0)

        # Recall the Poverty Line from the original data
        poverty_line = 53078.84


        # Calculate 'SP Ratio' as 'Salary' divided by 'Poverty Line'
        data_adjusted['SP Ratio'] = data_adjusted['Salary'] / poverty_line

        # Feature engineering for new fellows
        data_adjusted['Education_Combined'] = data_adjusted['Completion Status'] / 0.2
        data_adjusted['LivingStandards_Combined'] = data_adjusted['SP Ratio'] / 0.198
        data_adjusted['FoodSecurity_Combined'] = data_adjusted['SP Ratio'] / 0.2

        # Ensure 'Average Household Size' and 'Dependency Ratio' columns are present
        if 'Average Household Size' in data.columns and 'Dependency Ratio' in data.columns:
            data_adjusted['Average Household Size'] = np.random.choice(
                data['Average Household Size'],
                size=tot_fellows
            )
            data_adjusted['Dependency Ratio'] = np.random.choice(
                data['Dependency Ratio'],
                size=tot_fellows
            )

            # Calculate Rural Livelihoods Combined
            data_adjusted['RuralLivelihoods_Combined'] = (
                                                                 data_adjusted['SP Ratio'] - data_adjusted[
                                                             'Average Household Size'] - data_adjusted[
                                                                     'Dependency Ratio']
                                                         ) / 0.04

            # Calculate Risk Combined
            data_adjusted['Risk_Combined'] = data_adjusted['SP Ratio'] / 0.05
        else:
            st.error(
                "The 'Average Household Size' or 'Dependency Ratio' columns are not found in the original dataset.")

        # Step 7: Prepare features for prediction
        X_combined_adjusted = data_adjusted[['Education_Combined', 'LivingStandards_Combined',
                                             'FoodSecurity_Combined', 'RuralLivelihoods_Combined',
                                             'Risk_Combined']]

        # Step 8: Make predictions using the trained model
        try:
            adjusted_predictions = model.predict(X_combined_adjusted)
        except Exception as e:
            st.error(f"Error in making predictions: {e}")

        # Update metrics based on adjusted data
        # Scaled by Number of fellows from slider input
        # Average Household size in Nigeria is 5.06 persons per family
        total_persons_impacted = (len(data_adjusted[data_adjusted['SP Ratio'] >= 1]) * num_fellows * 5.06) / tot_fellows
        total_salary = data_adjusted['Salary'].sum() * num_fellows / tot_fellows
        avg_predicted_mpi = round(np.mean(adjusted_predictions), 3)
        avg_mpir = ((
                                previous_avg_actual_mpi - avg_predicted_mpi) / previous_avg_actual_mpi * 100)  # MPI Reduction Index

        # Create columns for displaying cards side by side
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # Display cards in the columns
        with col1:
            st.metric(
                label="Tot. Fellows",
                value=format_number(num_fellows),
                delta=format_number(num_fellows - previous_tot_fellows)
            )

        with col2:
            st.metric(
                label="Persons Impacted",
                value=format_number(total_persons_impacted),
                delta=format_number(total_persons_impacted - previous_total_persons_impacted)
            )

        with col3:
            st.metric(
                label="Total Salary",
                value=f"₦{format_number(total_salary)}",
                delta=f"₦{format_number(total_salary - previous_total_salary)}"
            )

        with col4:
            st.metric(
                label="Actual MPI",
                value=f"{previous_avg_actual_mpi:.3f}"
            )

        with col5:
            st.metric(
                label="Pred. MPI",
                value=f"{avg_predicted_mpi:.3f}",
                delta=f"{avg_predicted_mpi - previous_avg_predicted_mpi:.3f}"
            )

        with col6:
            st.metric(
                label="Reduction",
                value=f"{avg_mpir:.2f}%",
                delta=f"{avg_mpir - previous_avg_mpir:.2f}%"
            )

        # Plot Actual vs Predicted MPI
        # Create a DataFrame for actual and predicted MPI
        results_df = pd.DataFrame({'Actual MPI': y_test, 'Predicted MPI': y_pred})

        # Ensure index alignment with the original data's states
        if 'State of Residence' in data.columns:
            results_df['State of Residence'] = data['State of Residence'].iloc[y_test.index].values
        else:
            st.error("The 'State of Residence' column is not found in the original dataset.")

        # Step 1: Calculate average actual MPI values by state
        average_actual_mpi = data.groupby('State of Residence')['MPI'].mean().reset_index()
        average_actual_mpi.rename(columns={'MPI': 'Average Actual MPI'}, inplace=True)

        # Step 2: Calculate average adjusted predicted MPI values by state
        average_adjusted_predicted_mpi = pd.DataFrame({'State of Residence': data_adjusted['State of Residence'],
                                                           'Adjusted Predicted MPI': adjusted_predictions})
        average_adjusted_predicted_mpi = average_adjusted_predicted_mpi.groupby('State of Residence')[
                'Adjusted Predicted MPI'].mean().reset_index()

        # Step 4: Combine all into one DataFrame
        combined_mpi_df = average_actual_mpi.merge(average_adjusted_predicted_mpi, on='State of Residence', how='left')

        # Step 5: Set 'State of Residence' as the index for the line chart
        combined_mpi_df.set_index('State of Residence', inplace=True)

        # Step 6: Select only 'Adjusted Predicted MPI' and 'Actual MPI' for the line chart
        combined_mpi_selected_df = combined_mpi_df[['Average Actual MPI', 'Adjusted Predicted MPI']]

        # Step 6: Create the line chart using Streamlit
        # Step 6: Create the line chart using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined_mpi_selected_df.index,
            y=combined_mpi_selected_df['Average Actual MPI'],
            mode='lines',  # Line without markers
            line=dict(color='green', width=2),
            name='Average Actual MPI'
        ))

        # Add the Adjusted Predicted MPI line
        fig.add_trace(go.Scatter(
            x=combined_mpi_selected_df.index,
            y=combined_mpi_selected_df['Adjusted Predicted MPI'],
            mode='lines',  # Line without markers
            line=dict(color='#AFCF35', width=2),
            name='Adjusted Predicted MPI'
        ))

        # Customizing the layout
        fig.update_layout(
            title='MPI Values Comparison by State',
            title_font=dict(color='black', size=20),
            xaxis_title='State of Residence',
            yaxis_title='MPI Values',
            xaxis=dict(showline=True, linecolor='black', tickfont=dict(color='black')),
            yaxis=dict(showline=True, linecolor='black', tickfont=dict(color='black')),
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
            legend=dict(font=dict(color='black')),  # Legend text color black
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Calculate total number of households in and out of poverty based on SP Ratio and slider values
        total_households = (len(data_adjusted)) * num_fellows
        households_out_of_poverty = (len(data_adjusted[data_adjusted['SP Ratio'] >= 2]))*num_fellows
        households_out_of_poverty_refactor = households_out_of_poverty/100  # Refactored by data_size
        households_in_poverty = (total_households - households_out_of_poverty)/100

        # Metrics
        st.markdown(
            f"<h4 style='color: black; font-weight: bold;'>Households Poverty Status</h4>",
            unsafe_allow_html=True
        )

        col_hop, col_hip, col_lin = st.columns(3)
        with col_hop:
            custom_metric(
            label="Households no Poverty",
            value=f"{format_number(households_out_of_poverty_refactor)}",
                     )
        with col_hip:
            custom_metric(
            label="Households in Poverty",
            value=f"{format_number(households_in_poverty)}",
                    )
        with col_lin:
            custom_metric(
                label="Current Poverty Line",
                value=f"₦{poverty_line}",
            )

        # Doughnut chart showing households in poverty vs out of poverty
        household_data = {
                'Category': ['In Poverty', 'Out of Poverty'],
                'Count': [households_in_poverty, households_out_of_poverty_refactor]
            }

        # Create a dataframe for the household data
        df_household = pd.DataFrame(household_data)

        # Create columns for doughnut chart and table
        col_doughnut, col_table = st.columns(2)

        # Doughnut chart
        with col_doughnut:
            fig1 = go.Figure()

            # Add the doughnut chart trace
            fig1.add_trace(go.Pie(
                labels=df_household['Category'],
                values=df_household['Count'],
                hole=0.5,
                marker=dict(colors=['#AFCF35', 'green']),
                textinfo='percent'
            ))

            # Update layout to customize the chart's appearance
            fig1.update_layout(
                title='Households In vs Out of Poverty',
                title_font_color='black',
                title_font_size=20,
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for plot
                legend=dict(
                    title='Poverty Status',
                    title_font_color='black',
                    font=dict(color='black')  # Black legend text
                )
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig1, use_container_width=True)
        # Table (MPI Values by State) in the right column
        with col_table:
            st.write(" ")
            st.dataframe(combined_mpi_selected_df)

# Signature Footer
st.markdown(
    """
    <div style='display: flex; flex-direction: column; align-items: center; max-width: 400px; margin: 0 auto;'>
        <p style='color: #777777; text-align: center; font-weight: bold; font-size: 10px;'>Knowledge Showcase by Hon-Time: Bello Malik Pelumi FE/23/64160264</p>
    </div>
    """,
    unsafe_allow_html=True
)