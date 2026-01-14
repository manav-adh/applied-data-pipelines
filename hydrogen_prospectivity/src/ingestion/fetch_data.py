import rasterio
import matplotlib.pyplot as plt
import numpy as np



# Open the raster file
with rasterio.open('zip_files/02_SC_COS_unzipped/02_SC_COS/cos_sc.asc') as src:
    cos_data = src.read(1)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
print(cos_data.shape)

print(cos_data[:5, :5])
print("unique values in cos_data:", np.unique(cos_data))

masked_cos_data = np.ma.masked_where(cos_data == -9999, cos_data)

plt.figure(figsize=(10, 6))
plt.imshow(masked_cos_data, extent=extent, cmap='viridis')
plt.colorbar(label='Chance of Success (COS)')
plt.title('Chance Of Success Map')
plt.show()