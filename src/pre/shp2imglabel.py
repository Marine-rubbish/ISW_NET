import os
import numpy as np
import geopandas as gpd
from PIL import Image
from skimage.draw import polygon, line
from osgeo import gdal
import shapely

def segment_image(img, tag_img, tag_filename, img_seg_filename, chunk_size=(256, 256)):
    """
    对图像标签进行分割并保存包含标签像素(1)的图像。

    参数
    ---
    img : numpy.ndarray
        包含原始图像的三维数组。
    tag_img : numpy.ndarray
        包含图像标签的二维数组，标签区域为1，背景为0。
    img_filename : str, path object
        原始图像文件的路径。
    img_seg_path : str, path object
        保存分割后图像块的路径。
    tag_filename : str, path object
        生成的标签图像文件的保存路径。
    chunk_size : tuple, optional
        分割后图像块的大小，默认为 (256, 256)。
    
    功能
    ---
    将输入的图像标签分割为 256x256 大小的块，并保存包含标签区域的块。

    示例
    ---
    
    """
    # Divide tag_img into 256x256 blocks and pad edges
    height, width = tag_img.shape
    num_blocks_y = (height + 255) // chunk_size[0]
    num_blocks_x = (width + 255) // chunk_size[1]

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            start_y = by * chunk_size[0]
            start_x = bx * chunk_size[1]

            # Pad block if necessary
            pad_height = chunk_size[0] - (height - start_y)
            pad_width = chunk_size[1] - (width - start_x)
            
            if pad_height > 0 and pad_width > 0:
                block = tag_img[start_y - pad_height:start_y + block.shape[0], start_x - pad_width:start_x + block.shape[1]]
                block_image = img[:, start_y - pad_height:start_y + block.shape[0], start_x - pad_width:start_x + block.shape[1]]
            elif pad_height > 0:
                block = tag_img[start_y - pad_height:start_y + block.shape[0], start_x:start_x + chunk_size[1]]
                block_image = img[:, start_y - pad_height:start_y + block.shape[0], start_x:start_x + chunk_size[1]]
            elif pad_width > 0:
                block = tag_img[start_y:start_y + chunk_size[0], start_x - pad_width:start_x + block.shape[1]]
                block_image = img[:, start_y:start_y + chunk_size[0], start_x - pad_width:start_x + block.shape[1]]
            else:
                block = tag_img[start_y:start_y + chunk_size[0], start_x:start_x + chunk_size[1]]
                block_image = img[:, start_y:start_y + chunk_size[0], start_x:start_x + chunk_size[1]]

            

            # Save block if it contains any pixels with value 1
            if np.any(block == 1):
                try:
                    block_image = np.transpose(block_image, (1, 2, 0))

                    if block_image.shape[0] < chunk_size[0] or block_image.shape[1] < chunk_size[1]:
                        pass

                    block_image = Image.fromarray(block_image)
                    block_label_image = Image.fromarray(block)

                    block_image_filename = f"{os.path.splitext(img_seg_filename)[0]}_{by}_{bx}.png"
                    block_filename = f"{os.path.splitext(tag_filename)[0]}_{by}_{bx}.png"

                    block_image.save(block_image_filename)
                    block_label_image.save(block_filename)
                except Exception as e:
                    print(f"Error saving block: {e}")


def shp_to_image(shp_filename, img_filename, tag_filename, img_seg_filename):
    """
    将一个包含图像标签的 Shapefile 文件转换为一个图像文件。

    参数
    ----
    shp_filename : str, path object 或 file-like object
        Shapefile 文件的路径，可以是绝对路径或相对路径，也可以是具有 `read()` 方法的对象（如打开的文件或 StringIO）。
    img_filename : str, path object 或 file-like object
        用于获取地理变换信息的 GeoTIFF 图像文件的路径。
    tag_filename : str, path object
        生成的标签图像文件的保存路径。
    img_seg_filename : str, path object
        保存分割后图像块的路径。

    示例
    ----
    ```python
    shp_to_image("path/to/shapefile.shp", "path/to/image.tif", "path/to/output.png")
    ```

    注意
    ----
    - 该函数依赖于以下库：`geopandas`, `numpy`, `gdal`, `shapely`, `matplotlib` 等。
    - 确保输入的 Shapefile 和图像文件具有一致的坐标参考系统（CRS）。
    - 如果 Shapefile 包含多种几何类型（如 Polygon 和 LineString），函数会分别处理它们并绘制在同一图像上。
    - 输出的图像标签区域为1，背景为0。
    """

    # Read the shapefile
    gdf = gpd.read_file(shp_filename)

    # Open the GeoTIFF image
    dataset = gdal.Open(img_filename)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open {img_filename}")
    image_size = (dataset.RasterYSize, dataset.RasterXSize)
    img = dataset.ReadAsArray()
    
    # Create an empty image
    tag_img = np.zeros(image_size, dtype=np.uint8)

    # Get geotransform from the dataset
    geotransform = dataset.GetGeoTransform()
    inv_geotransform = gdal.InvGeoTransform(geotransform)

    def world_to_pixel(x, y):
        px, py = gdal.ApplyGeoTransform(inv_geotransform, x, y)
        return int(round(px)), int(round(py))

    # Transform shapefile geometries to pixel coordinates
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: shapely.ops.transform(
        lambda x, y: world_to_pixel(x, y), geom))
    
    # # Get the bounds of the shapefile
    # minx, miny, maxx, maxy = gdf.total_bounds
    
    # # Calculate the scaling factors
    # x_scale = image_size[1] / (maxx - minx)
    # y_scale = image_size[0] / (maxy - miny)
    
    # Iterate over the geometries in the shapefile
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        
        if geom.geom_type == 'Polygon':
            # TODO: Handle MultiPolygon
            # Get the coordinates of the polygon
            coords = np.array(geom.exterior.coords)
            
            # Scale the coordinates to the image size
            coords[:, 0] = (coords[:, 0] - minx) * x_scale
            coords[:, 1] = (maxy - coords[:, 1]) * y_scale
            
            # Draw the polygon on the image
            rr, cc = polygon(coords[:, 1], coords[:, 0], img.shape)
            img[rr, cc] = 255
        
        elif geom.geom_type == 'LineString':
            # Get the coordinates of the line
            coords = np.array(geom.coords)
            # print(coords)
            
            # # Scale the coordinates to the image size
            # coords[:, 0] = (coords[:, 0] - minx) * x_scale
            # coords[:, 1] = (maxy - coords[:, 1]) * y_scale
            
            # Mark each point on the image
            for coord in coords:
                x, y = int(coord[0]), int(coord[1])
                if 0 <= y < tag_img.shape[0] and 0 <= x < tag_img.shape[1]:
                    tag_img[y, x] = 1

    # Segment the image
    segment_image(img, tag_img, tag_filename, img_seg_filename)

    # Save the image
    # Image.fromarray(tag_img).save(output_filename)
    # print(f"Saved {output_filename}")

def process_directory(shp_dir, output_dir, img_dir, img_seg_dir):
    """
    处理给定目录中的所有 .shp 文件，并将其转换为图像文件。
    如果输出目录不存在，则创建该目录。

    参数
    ---
    shp_dir (str): 包含 .shp 文件的目录路径。

    output_dir (str): 保存转换后图像文件的目录路径。

    img_dir (str): 保存原始图像文件的目录路径。

    img_seg_dir (str): 保存分割后图像块的目录路径。

    返回
    ---
    None

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(shp_dir):
        if filename.endswith('.shp'):
            shp_filename = os.path.join(shp_dir, filename)
            img_path = os.path.join(img_dir, os.path.splitext(filename)[0] + '.tiff')
            img_filename = os.path.join(os.path.dirname(img_path), 'MODIS_TrueColor_' + os.path.basename(img_path)[3:])
            img_seg_filename = os.path.join(img_seg_dir, 'MODIS_TrueColor_' + os.path.basename(img_path)[3:])
            output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
            shp_to_image(shp_filename, img_filename, output_filename, img_seg_filename)

if __name__ == "__main__":
    shp_dir = r'E:\内波数据集-大数据中心上传\ISW_product_20240325'
    output_dir = r'E:\内波数据集-大数据中心上传\IW_imgaes_label'
    img_dir = r'E:\内波数据集-大数据中心上传\IW_images'
    img_seg_dir = r'E:\内波数据集-大数据中心上传\IW_images_seg'
    process_directory(shp_dir, output_dir, img_dir, img_seg_dir)