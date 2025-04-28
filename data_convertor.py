import os
import shutil
import numpy as np
import pandas as pd

# # Paths
# original_csv = 'JaundiceDataDesktop_2\JaundiceRecords.xlsx'
# # Update with your CSV filename
# dataset_root = 'JaundiceDataDesktop_2\Dataset\Dataset'
# output_img_folder = 'JaundiceDataDesktop_2\SimplifiedImages'
# output_csv = 'JaundiceDataDesktop_2\simplified_data.csv'

# # Read your CSV
# df = pd.read_excel(original_csv)

# # Function to extract patient folder name from image path in CSV
# def extract_folder_name(image_path):
#     # Normalize path separators and get the folder just above the file
#     return os.path.basename(os.path.dirname(image_path))

# # Add a column with patient folder name
# df['patient_folder'] = df['Forehead-0'].apply(extract_folder_name)  # Replace 'image_path_column' with your actual column name

# # Prepare output folder
# os.makedirs(output_img_folder, exist_ok=True)

# # Prepare list for new CSV rows
# new_rows = []

# # Loop through each patient folder
# count=77
# for patient_folder in os.listdir(dataset_root):
#     folder_path = os.path.join(dataset_root, patient_folder)
#     if not os.path.isdir(folder_path):
#         continue

#     # Get all rows in the CSV for this patient
#     rows = df[df['patient_folder'] == patient_folder]
#     if rows.empty:
#         continue
#     count+=1
#     # Loop through images in this folder
#     for img_file in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_file)
#         if not os.path.isfile(img_path):
#             continue

#         # You can choose how to rename images, here we keep original name prefixed by patient_folder
#         new_img_name = f"{count}_{img_file}"
#         new_img_path = os.path.join(output_img_folder, new_img_name)
#         shutil.copy2(img_path, new_img_path)

#         # For each image, create a row in the new CSV for each matching patient row
#         for _, row in rows.iterrows():
#             row_dict = row[['Weight (g)', 'Gender', 'Treatment','Jaundice Decision', 'Postnatal Age (days)']]
#             row_dict['new_image_name'] = new_img_name
#             new_rows.append(row_dict)

# # Create new DataFrame and save
# new_df = pd.DataFrame(new_rows)
# new_df.to_csv(output_csv, index=False)

# dataset_path= 'JaundiceDataDesktop_1\Dataset'
# count=0
# for folder in os.listdir(dataset_path):
#     folder_path = os.path.join(dataset_path, folder)
#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         count+=1
#     print(count)

# csv_file_path = 'JaundiceDataDesktop_1\simplified_data.csv'
# columns_to_extract = ['new_image_name', 'Postnatal Age (hrs)', 'Weight (g)', 'Treatment', 'Jaundice Decision', ]
# df = pd.read_csv(csv_file_path)
# extracted_df = df[columns_to_extract]
# output_csv = 'JaundiceDataDesktop_1\extracted_data.csv'
# extracted_df.to_csv(output_csv, index=False)

# csv_file1_path = 'JaundiceDataDesktop_1\extracted_data.csv'
# csv_file2_path = 'JaundiceDataDesktop_2\extracted_data.csv'

# output_csv_path = 'data\clinical_data.csv'

# df1 = pd.read_csv(csv_file1_path)
# df2 = pd.read_csv(csv_file2_path)
# result = pd.concat([df1, df2], ignore_index=True)
# result.to_csv(output_csv_path, index=False)

# def map_values (val):
#     if val == 'Phototherapy':
#         return 1
#     else :
#         return '0'
    
# csv_file_path= 'data/clinical_data.csv'
# df = pd.read_csv(csv_file_path)
# df['Treatment'] = df['Treatment'].apply(map_values)

# df.to_csv(csv_file_path, index=False)

# csv_file_path = 'data/clinical_data.csv'
# df = pd.read_csv(csv_file_path)

# col = df['Weight (g)']
# mean_val = col.mean()

# df.loc[df['Weight (g)'] < 500, 'Weight (g)'] = mean_val.round(0)

# output_csv_path = 'data/clinical_data_updated.csv'
# df.to_csv(output_csv_path, index=False)

csv_file_path = 'data/clinical_data_updated.csv'
df = pd.read_csv(csv_file_path)
df['Postnatal Age (hrs)'] = df['Postnatal Age (hrs)'].astype(str).str.extract(r'(\d+)')[0].astype(float)
df['Postnatal Age (hrs)'] = pd.to_numeric(df['Postnatal Age (hrs)'], errors='coerce')

output_csv_path = 'data/clinical_data_updated.csv'
df.to_csv(output_csv_path, index=False)