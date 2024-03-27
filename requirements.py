import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go  # No need to import plotly.graph_objects again
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Set page configuration
# Set page configuration
st.set_page_config(page_title="Data Visualization App", layout="wide")

col_header1, col_header2, col_header3 = st.columns([3, 4, 0.6])

with col_header3:
    st.write("Antero Eng Tool")

# Custom CSS for bottom border
st.markdown(
    """
    <style>
    .chart-container {
        border-bottom: 2px solid #ccc;
        padding-bottom: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize data variable
data = None

# CSV Upload and Dynamic Chart Generation
def main():
    global data

    st.title("Data Visualization and Analysis App")

    # CSV file upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV data
        data = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")

        # Data overview
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.markdown("---")
        st.subheader("Data Overview")
        
        col_dataoverview1, col_dataoverview2, col_dataoverview3 = st.columns([0.3, 0.3, 4])

        with col_dataoverview1:
            st.write(f"**Total Rows:** {data.shape[0]}")
        with col_dataoverview2:
            st.write(f"**Total Columns:** {data.shape[1]}")

        columns = data.columns.tolist()
        selected_columns = st.multiselect("Please choose the columns you wish to display in table format, or skip this section if you prefer not to generate a table.", columns)

        # Interactive table generation
        if selected_columns:
            num_rows = st.number_input("Number of rows to display", min_value=1, value=10)
            st.dataframe(data[selected_columns].head(num_rows))

        # Dynamic Chart Generation
        st.subheader("")
        st.subheader("")
        st.subheader("")
        st.markdown("---")
        st.subheader("Dynamic Chart Generation")

        # Adding a new column "Number/Duration"
        if data is not None:
            data['Row Number'] = range(len(data))
            columns = data.columns.tolist()

            # Configure charts
            num_charts = st.number_input("Number of charts to generate", min_value=1, max_value=10, value=1)
            st.write("")
            st.write("")
            
            # Input fields for x-axis range for all charts
            x_min = data['Row Number'].min()
            x_max = data['Row Number'].max()

            
            st.write("Select X-axis range for all charts:")
            col_x_range_start, col_x_range_end, col_remain = st.columns([0.6, 0.6, 5])
            with col_x_range_start:
                start_x_all = st.number_input("Enter X-axis start", min_value=x_min, max_value=x_max, value=x_min)
            with col_x_range_end:
                end_x_all = st.number_input("Enter X-axis end", min_value=x_min, max_value=x_max, value=x_max)
            st.write("")
            st.write("")

            summary_data = []  # Initialize a list to store summary values

            for i in range(num_charts):
                st.subheader(f"Chart {i+1}")

                col1, col_chart, col2 = st.columns([1, 3, 1])

                with col1:
                    chart_type = st.selectbox(f"Select chart type for Chart {i+1}", ["Line", "Bar", "Scatter"], key=f"chart_type_{i}")
                    
                    # Initialize warning message
                    warning_message = st.empty()
                    
                    x_column = st.selectbox(f"Select X-axis column for Chart {i+1}", columns, index=columns.index("Row Number"))
                    y_columns = st.multiselect(f"Select Y-axis column(s) for Chart {i+1}", columns)

                    # Check if both X and Y values are provided
                    if len(x_column) > 0 and len(y_columns) > 0:
                        # If both X and Y values are provided, remove the warning message
                        warning_message.empty()
                    else:
                    # If either X or Y values are missing, display the warning message
                        warning_message.warning("Please input X and Y values with NUMBER format, not DATE or TEXT.")

                    # Check if X and Y values are provided and the chart type is not Bar
                    if chart_type != "Bar" and len(x_column) > 0 and len(y_columns) > 0:
                        trendline = st.checkbox(f"Add trendline for Chart {i+1}. **Only for LINE & SCATTER CHART TYPE!**", key=f"trendline_{i}")
                    else:
                        trendline = False

                    if trendline:
                        col_a, col_b = st.columns([1.5, 1.5])
                        with col_a:
                            trendline_type = st.selectbox(f"Select trendline type", ["Linear", "Average", "Polynomial"], key=f"trendline_type_{i}")
                        with col_b:
                            if trendline_type == "Polynomial":
                                degrees = {}
                                for y_column in y_columns:
                                    degrees[y_column] = st.number_input(f"Degree for {y_column}", min_value=2, max_value=20, value=2, key=f"degree_{i}_{y_column}")
                    st.subheader("")

                with col_chart:
                    with st.container():
                        if x_column and y_columns:
                            filtered_data = data[(data['Row Number'] >= start_x_all) & (data['Row Number'] <= end_x_all)]

                            if chart_type == "Line":
                                fig = px.line(filtered_data, x=x_column, y=y_columns)
                            elif chart_type == "Bar":
                                fig = px.bar(filtered_data, x=x_column, y=y_columns)
                            else:
                                fig = px.scatter(filtered_data, x=x_column, y=y_columns)

                            if trendline:
                                for trace in fig.data:
                                    if trace.name in y_columns:
                                        color = trace.line.color if hasattr(trace.line, 'color') else trace.marker.color
                                        if trendline_type == "Linear":
                                            slope, intercept, _, _, _ = linregress(filtered_data[x_column], filtered_data[trace.name])
                                            fig.add_shape(type="line", x0=start_x_all, y0=slope*start_x_all+intercept,
                                                        x1=end_x_all, y1=slope*end_x_all+intercept,
                                                        line=dict(color=color, width=2, dash="dash"))
                                            y_predicted = slope * filtered_data[x_column] + intercept
                                            r_squared = r2_score(filtered_data[trace.name], y_predicted)
                                        elif trendline_type == "Average":
                                            avg_y_value = filtered_data[trace.name].mean()
                                            fig.add_hline(y=avg_y_value, line_dash="dash",
                                                        annotation_text=f"Avg {trace.name} = {avg_y_value:.5f}",
                                                        annotation_position="bottom right",
                                                        line=dict(color=color)) 
                                        elif trendline_type == "Polynomial":
                                            degree = degrees[trace.name]
                                            coeffs = np.polyfit(filtered_data[x_column], filtered_data[trace.name], degree)
                                            poly_function = np.poly1d(coeffs)
                                            equation = " + ".join(f"{coeffs[i]:.8f} * x^{degree-i}" for i in range(degree+1))
                                            x_values = np.linspace(start_x_all, end_x_all, 100)
                                            y_values = poly_function(x_values)
                                            r_squared = r2_score(filtered_data[trace.name], poly_function(filtered_data[x_column]))
                                            fig.add_trace(go.Scatter(x=x_values, y=y_values, line_dash="dash",
                                                                    name=f"Polynomial Trendline {degree} for {trace.name}",
                                                                    line=dict(color=color))) 

                            fig.update_layout(title=f"Chart {i+1}", xaxis_title=x_column, yaxis_title="Value", height=400) 
                            st.plotly_chart(fig, use_container_width=True)
                            

                with col2:
                    st.write(f"**Chart {i+1} Summary:**")
                    st.write("")
                    st.write("")

                    for y_column in y_columns:
                        st.write("• **Min of**", y_column + ":", f"{filtered_data[y_column].min():.5f}")
                        st.write("• **Max of**", y_column + ":", f"{filtered_data[y_column].max():.5f}")
                        st.write("• **Average of**", y_column + ":", f"{filtered_data[y_column].mean():.5f}")
                        st.write("• **Standard Deviation of**", y_column + ":", f"{filtered_data[y_column].std():.5f}")
                
                
                
                # Append the chart summary data to the summary_data list
                chart_summary_data = []

                for y_column in y_columns:
                    data_row = {
                        "Chart": f"Chart {i+1}",
                        "Y Column": y_column,
                        "Min Value": filtered_data[y_column].min(),
                        "Max Value": filtered_data[y_column].max(),
                        "Average Value": filtered_data[y_column].mean(),
                        "Standard Deviation": filtered_data[y_column].std()
                    }

                    if trendline:
                        if trendline_type == "Linear":
                            data_row["Trendline Equation"] = f"y = {slope:.5f}x + {intercept:.5f}"
                            data_row["R-squared Value"] = f"{r_squared:.5f}"
                        elif trendline_type == "Polynomial":
                            data_row["Trendline Equation"] = f"y = {equation}"
                            data_row["R-squared Value"] = f"{r_squared:.5f}"

                    chart_summary_data.append(data_row)

                summary_data.extend(chart_summary_data)  # Extend the summary data with the chart summary data

            # Convert the summary data into a DataFrame
            summary_df = pd.DataFrame(summary_data)

            # Display the summary DataFrame with automatic width adjustment
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.markdown("---")
            st.subheader("Summary Table")
            st.dataframe(summary_df, use_container_width=True)


# Run the app
if __name__ == "__main__":
    main()
