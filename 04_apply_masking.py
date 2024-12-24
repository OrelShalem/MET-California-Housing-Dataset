# 04_apply_masking.py

# pandas is used to load the data
# numpy is used to mask the data
import pandas as pd
import numpy as np

# Load the encoded data
df = pd.read_csv('data/encoded_data.csv')

# Define the masking function
def mask_data(df, mask_fraction=0.15):
    """
    מיסוך נתונים בגישת MET:
    1. מיסוך אחיד לכל התכונות
    2. אחוז מיסוך נמוך יותר
    3. שימוש בערך מיסוך אחיד (-1)
    """
    df_masked = df.copy()
    
    # הגדרת כל התכונות ביחד
    features = [
        'MedInc', 'AveRooms', 'AveBedrms', 'Population', 
        'AveOccup', 'Latitude', 'Longitude', 'AgeCategory'
    ]
    
    # יצירת מסיכה אקראית לכל התכונות
    total_cells = len(df) * len(features)
    num_masks = int(total_cells * mask_fraction)
    
    # בחירת תאים אקראיים למיסוך
    mask_indices = []
    for _ in range(num_masks):
        row = np.random.randint(0, len(df))
        col = np.random.choice(features)
        mask_indices.append((row, col))
    
    # החלת המיסוך
    for row, col in mask_indices:
        df_masked.loc[row, col] = -1  # ערך מיסוך אחיד
    
    # שמירת אינדקסים של התאים הממוסכים
    mask_info = pd.DataFrame(mask_indices, columns=['row', 'column'])
    
    return df_masked, mask_info

# Apply masking
df_masked, mask_info = mask_data(df)

# Save the masked data and mask indices
df_masked.to_csv('data/masked_data.csv', index=False)
mask_info.to_csv('data/mask_info.csv', index=False)

print('Data masked and saved to data/masked_data.csv')
print(f'Total cells masked: {len(mask_info)}')
print(f'Masking percentage: {(len(mask_info) / (len(df) * 8)) * 100:.2f}%')