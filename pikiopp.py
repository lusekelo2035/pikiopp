import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt


st.set_page_config(page_title="pikiopp",page_icon="🛵")

theme_plotly = None 

# Define custom CSS style
custom_css = """
<style>
/* Adjust font size and family */
body {
    font-family: Arial, sans-serif;
    font-size: 14px; /* Adjust default font size */
}

/* Center align text */
h1, h2, h3, h4, h5, h6 {
    text-align: center;
    font-size: 24px; /* Adjust heading font size */
}

/* Add padding to sections */
.section {
    padding: 20px;
}

/* Style table */
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px; /* Adjust table font size */
}

table th, table td {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 12px; /* Increase cell padding */
}

table th {
    background-color: #f2f2f2;
    font-size: 16px; /* Adjust table heading font size */
}

/* Style sidebar */
.sidebar .sidebar-content {
    background-color: #f0f0f0;
    padding: 20px;
}

/* Adjust width of sidebar */
.sidebar .sidebar-content .block-container {
    max-width: 350px;
}
</style>
"""

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Main Streamlit app code...



# Function to calculate distance using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in km

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Function to classify delivery fee as below, above, or within standard
def classify_delivery_fee(distance, delivery_fee):
    if 0 <= distance <= 1:
        standard_fee = 2500
    elif 1 <= distance <= 3:
        standard_fee = 3000
    elif 3 <= distance <= 5:
        standard_fee = 4000
    elif 5 <= distance <= 7:
        standard_fee = 5000
    elif 6 <= distance <= 8:
        standard_fee = 6000
    elif 8 <= distance <= 10:
        standard_fee = 8000
    elif 10 <= distance <= 12:
        standard_fee = 10000
    elif 12 <= distance <= 15:
        standard_fee = 12000
    else:
        standard_fee = None
    
    if standard_fee is not None:
        if delivery_fee < standard_fee:
            return "Below"
        elif delivery_fee > standard_fee:
            return "Above"
        else:
            return "Within"
    else:
        return None


standard_delivery_fee = {
    'Distance': ['0-5 km', '1-3 km', '3-5 km', '5-7 km', '6-8 km', '8-10 km', '10-12 km', '12-15 km'],
    'Amount': [2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000]
}

# Create DataFrame
df_standard_delivery_fee = pd.DataFrame(standard_delivery_fee)

# Interface layout
st.title("Welcome to Our Delivery Service aanalysis")

# Present standard delivery fee in a table
st.subheader("Standard Delivery Fee:")
st.write(df_standard_delivery_fee)

# Plot standard delivery fee range
st.subheader("Standard Delivery Fee Range:")
plt.figure(figsize=(10, 6))
plt.plot(df_standard_delivery_fee['Distance'], df_standard_delivery_fee['Amount'], marker='o')
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Fee (TSh)")
plt.title("Standard Delivery Fee Range")
plt.grid(True)
plt.xticks(rotation=45)
st.pyplot(plt)


# Main Streamlit app code
def main():
    st.title("feel free to upload data file for analysis")

    # File upload section
    upload = st.file_uploader("Upload Your Dataset (In CSV or Excel Format)", type=["csv", "xlsx"])

    if upload is not None:
        try:
            # Check the file type and read the data
            if upload.type == 'application/vnd.ms-excel':
                data = pd.read_excel(upload, engine='openpyxl')
            else:
                data = pd.read_csv(upload)

            # Calculate distances and add them to the DataFrame
            distances = []
            for index, row in data.iterrows():
                distance = haversine(
                    row['CUSTOMER LATITUDE'],
                    row['CUSTOMER LONGITUDE'],
                    row['BUSINESS LATITUDE'],
                    row['BUSINESS LONGITUDE']
                )
                distances.append(distance)

            data['DISTANCE (km)'] = distances

            # Calculate delivery fee classification
            delivery_fee_classification = []
            for index, row in data.iterrows():
                classification = classify_delivery_fee(row['DISTANCE (km)'], row['DELIVERY FEE'])
                delivery_fee_classification.append(classification)

            data['DELIVERY FEE CLASSIFICATION'] = delivery_fee_classification
            
            
            # Summary of delivery fee classification for all orders
            st.subheader("Summary of Delivery Fee Classification for All Orders")
            classification_counts_all = data['DELIVERY FEE CLASSIFICATION'].value_counts()
            percentages_all = (classification_counts_all / len(data)) * 100
            percentages_all = percentages_all.round(1)  # Round percentages to one decimal place
            
            # Include "None" classification
            none_count = len(data[data['DELIVERY FEE CLASSIFICATION'].isnull()])
            none_percentage = (none_count / len(data)) * 100
            classification_counts_all['None'] = none_count
            percentages_all['None'] = none_percentage
            
            summary_df = pd.DataFrame({'Count': classification_counts_all, 'Percentage': percentages_all})
            st.write(summary_df)
            
            # Plotting
            plt.figure(figsize=(8, 6))
            percentages_all.plot(kind='bar', color='skyblue')
            plt.title('Percentage of Orders by Delivery Fee Classification')
            plt.xlabel('Delivery Fee Classification')
            plt.ylabel('Percentage')
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            # Display the plot in Streamlit
            st.pyplot(plt)

            # Table to count the total number of orders in each category of delivery fee
            st.subheader("Total Number of Orders by Delivery Fee Category")
            
            # Define the distance and corresponding standard fees
            distance_ranges = [
                (0, 1),        
                (1, 3),        
                (3, 5),        
                (5, 7),        
                (6, 8),        
                (8, 10),       
                (10, 12),
                (12, 15)       
            ]
            
            standard_fees = [2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000]
            
            # Initialize counters for each category
            category_counts = {f"{low}-{high} km": {'Count': 0, 'Delivery Fee Amount': standard_fee} for (low, high), standard_fee in zip(distance_ranges, standard_fees)}
            category_counts['None'] = {'Count': 0, 'Delivery Fee Amount': None}  # Initialize count for "None" category
            
            # Count orders in each category
            for index, row in data.iterrows():
                distance = int(round(row['DISTANCE (km)']))  # Round distance and convert to integer
                delivery_fee = row['DELIVERY FEE']
                for (low, high), standard_fee in zip(distance_ranges, standard_fees):
                    if low <= distance < high and delivery_fee == standard_fee:
                        category_counts[f"{low}-{high} km"]['Count'] += 1
                if not any(low <= distance < high for (low, high) in distance_ranges):
                    category_counts['None']['Count'] += 1
            
            # Convert counts to DataFrame
            counts_df = pd.DataFrame.from_dict(category_counts, orient='index')
            counts_df.index.name = 'Distance Range'
            
            # Calculate percentages and add "%"
            total_orders = counts_df['Count'].sum()
            counts_df['Percentage'] = (counts_df['Count'] / total_orders) * 100
            counts_df['Percentage'] = counts_df['Percentage'].astype(int).astype(str) + '%'
            
            # Reorder the columns
            counts_df = counts_df[['Delivery Fee Amount', 'Count', 'Percentage']]
            
            # Display the table
            st.write(counts_df)


      


            # Filter by delivery fee range
            st.sidebar.subheader("Filter Orders by Delivery Fee Range")
            min_fee = st.sidebar.number_input("Minimum Delivery Fee", value=0, step=100)
            max_fee = st.sidebar.number_input("Maximum Delivery Fee", value=int(data['DELIVERY FEE'].max()), step=100)

            if min_fee <= max_fee:
                filtered_by_fee = data[(data['DELIVERY FEE'] >= min_fee) & (data['DELIVERY FEE'] <= max_fee)]
                if not filtered_by_fee.empty:
                    st.subheader(f"Orders with Delivery Fee Range: {min_fee} to {max_fee}")
                    filtered_by_fee = filtered_by_fee[['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']]
                    st.write(filtered_by_fee)
                else:
                    st.write("No orders found within the specified delivery fee range.")
            else:
                st.error("Please ensure that the minimum delivery fee is less than or equal to the maximum delivery fee.")

            # Filter by restaurant name, ID, and customer address
            st.sidebar.subheader("Filter Orders by Restaurant Name, ID, or Customer Address")
            filter_option = st.sidebar.selectbox("Filter by:", ["Restaurant Name", "ID", "Customer Address"])
            search_query = st.sidebar.text_input(f"Search by {filter_option}:")
            if search_query:
                if filter_option == "Restaurant Name":
                    filtered_data = data[data['BUSINESS NAME'].str.contains(search_query, case=False)]
                elif filter_option == "ID":
                    filtered_data = data[data['ID'] == int(search_query)]
                elif filter_option == "Customer Address":
                    filtered_data = data[data['CUSTOMER ADDRESS'].str.contains(search_query, case=False)]

                if not filtered_data.empty:
                    st.subheader(f"Filtered Data for '{search_query}'")
                    filtered_data = filtered_data[['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']]
                    st.write(filtered_data)
                else:
                    st.write(f"No matching records found for '{search_query}'.")
                    
                              

            # Sidebar filter to check standard delivery fee classification
            st.sidebar.subheader("Check Standard Delivery Fee Classification")
            standard_fee_classification = st.sidebar.selectbox("Select Classification:", ["Below", "Within", "Above", "None"])
            
            if standard_fee_classification:
                st.subheader(f"Selected Classification: {standard_fee_classification}")
            
                if standard_fee_classification == "None":
                    filtered_by_standard = data[data['DELIVERY FEE CLASSIFICATION'].isnull()]
                else:
                    filtered_by_standard = data[data['DELIVERY FEE CLASSIFICATION'] == standard_fee_classification]
            
                st.write(f"Filtered Data Length: {len(filtered_by_standard)}")
            
                st.subheader(f"Orders with Standard Delivery Fee Classification: {standard_fee_classification}")
            
                # Display the filtered data in a table
                columns_to_display = ['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']
                st.write(filtered_by_standard[columns_to_display])
                            


        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

