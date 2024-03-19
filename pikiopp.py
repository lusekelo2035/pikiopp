import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

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

# Main Streamlit app code
def main():
    st.title("Delivery Data Analysis App")

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

            # Display top orders with highest delivery fee
            st.subheader("Top Orders with Highest Delivery Fee")
            top_orders_fee = data.nlargest(5, 'DELIVERY FEE')[['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']]
            st.write(top_orders_fee)

            # Display top orders with highest distance
            st.subheader("Top Orders with Highest Distance")
            top_orders_distance = data.nlargest(5, 'DISTANCE (km)')[['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']]
            st.write(top_orders_distance)

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
            standard_fee_classification = st.sidebar.selectbox("Select Classification:", ["Below", "Within", "Above"])
            
            if standard_fee_classification:
                st.write(f"Selected Classification: {standard_fee_classification}")
            
                filtered_by_standard = data[data['DELIVERY FEE CLASSIFICATION'] == standard_fee_classification]
            
                st.write(f"Filtered Data Length: {len(filtered_by_standard)}")
            
                st.subheader(f"Orders with Standard Delivery Fee Classification: {standard_fee_classification}")
            
                # Display the filtered data in a table
                columns_to_display = ['ID', 'BUSINESS NAME', 'CUSTOMER NAME', 'CUSTOMER ADDRESS', 'DELIVERY FEE', 'DISTANCE (km)', 'DELIVERY FEE CLASSIFICATION']
                st.write(filtered_by_standard[columns_to_display])
            

            
            # Summary of delivery fee classification for all orders
            st.subheader(" Summary of Delivery Fee Classification for All Orders")
            classification_counts_all = data['DELIVERY FEE CLASSIFICATION'].value_counts()
            percentages_all = (classification_counts_all / len(data)) * 100
            percentages_all = percentages_all.round(1)  # Round percentages to one decimal place
            summary_df = pd.DataFrame({'Count': classification_counts_all, 'Percentage': percentages_all})
            st.write(summary_df)

            # Calculate percentages
            total_orders_all = len(data)
            percentages_all = (classification_counts_all / total_orders_all) * 100
            
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
                


        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

