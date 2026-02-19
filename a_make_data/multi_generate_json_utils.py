import os
import numpy as np
import utm
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import os
import pandas as pd
import re
from scipy.interpolate import interp1d
import requests

class OxtData:
    def __init__(self, lat, lon, alt, roll, pitch, yaw, pos_accuracy):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.pos_accuracy = pos_accuracy

def read_oxt_file(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        data = list(map(float, line.split()))
        # Assuming the order of data is: lat, lon, alt, roll, pitch, yaw, pos_accuracy
        oxt_data = OxtData(*data[:7])
    return oxt_data

# 读取 OXTS 数据文件，提取经纬度值
def read_oxts_files(file_paths):
    all_coords = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            line = file.readline()
            data = list(map(float, line.split()))
            # 假设数据顺序为：lat, lon, alt, roll, pitch, yaw, pos_accuracy
            lat, lon = data[0], data[1]
            # print(f"lat: {lat}, lon: {lon}")
            all_coords.append((lon, lat))  # 注意这里经纬度的顺序
    return all_coords

def convert_to_utm(oxt_data):
    # Convert latitude and longitude to UTM coordinates
    x, y, zone_number, zone_letter = utm.from_latlon(oxt_data.lat, oxt_data.lon)
    return x, y, zone_number, zone_letter

def calculate_distance(p1, p2):
    # Calculate Euclidean distance between two points
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_oxts_folder(folder_path, start_index=22, distance_threshold=2.0):
    utm_pose = []
    last_utm = None
    utm_to_info = {}

    # Sort the files by their numerical part
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')], key=lambda x: int(x.split('.')[0]))

    for i, filename in enumerate(files):
        if i < start_index:
            continue
        
        file_path = os.path.join(folder_path, filename)
        oxt_data = read_oxt_file(file_path)
        x, y, zone_number, zone_letter = convert_to_utm(oxt_data)
        
        if last_utm is None or calculate_distance(last_utm, (x, y)) >= distance_threshold:
            utm_pose.append((x, y, zone_number, zone_letter))
            last_utm = (x, y)
            # Record the UTM coordinates, latitude, longitude, and filename
            utm_to_info[(x, y, zone_number, zone_letter)] = {
                'lat': oxt_data.lat,
                'lon': oxt_data.lon,
                'yaw': oxt_data.yaw,
                'filename': filename
            }

    return utm_pose, utm_to_info

def calculate_relative_positions(utm_pose, num_points=15):
    trajectory_ins = {}
    
    for i, current_point in enumerate(utm_pose):
        if i + num_points > len(utm_pose):
            # If there are fewer than num_points points after the current point, skip it
            continue
        
        relative_positions = []
        for j in range(i, i + num_points):
            next_point = utm_pose[j]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            relative_positions.append((-dx, -dy))  # 注意转换坐标系
        
        # Store the relative positions in the dictionary
        trajectory_ins[tuple([current_point[0],current_point[1]])] = relative_positions
    
    return trajectory_ins

def calculate_past_relative_positions(utm_pose, num_points=5):
    trajectory_ins_past = {}
    
    for i, current_point in enumerate(utm_pose):
        if i < num_points:
            print("If there are fewer than num_points points before the current point, skip it")
            print("i",i)
            print("num_points",num_points)

            continue
        
        relative_positions = []
        for j in range(i - 1, max(i - num_points - 1, -1), -1):
            prev_point = utm_pose[j]
            dx = prev_point[0] - current_point[0]
            dy = prev_point[1] - current_point[1]
            relative_positions.append((-dx, -dy)) # 注意转换坐标系
        
        # Store the relative positions in the dictionary
        trajectory_ins_past[tuple([current_point[0],current_point[1]])] = relative_positions
        # print("current_point",tuple([current_point[0],current_point[1]]))

    
    return trajectory_ins_past
# Function to get the latitude, longitude, and filename for a given UTM coordinate
def get_info_for_utm(utm_coord, utm_to_info):
    return utm_to_info.get(utm_coord, None)

# 将unique_a转换为utm
def gps_to_utm(lat, lon):
    """
    Convert GPS coordinates (latitude, longitude, altitude) to UTM coordinates.

    Parameters:
    - lat: float, latitude in degrees
    - lon: float, longitude in degrees
    - alt: float, altitude in meters

    Returns:
    - x: float, UTM x-coordinate
    - y: float, UTM y-coordinate
    """
    x, y, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return x, y

def cal_interp1d(translations):
    # 给定的点集合
    points = translations
    # 计算每个点之间的距离
    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])
        distances.append(dist)
    cumulative_distances = np.cumsum(distances)

    # 创建插值函数
    x = points[:, 0]
    y = points[:, 1]

    print("cumula",cumulative_distances)
    print("x",x)
    # interp1d 要求输入数组至少要包含两个元素，所以我添加一个逻辑，当输入数组只有一个元素时，返回一个flag
    # 当确认为该flag时，跳过该层循环
    # 检查输入数据是否有足够的点
    if len(cumulative_distances) < 2 or len(x) < 2:
        print("输入数据不足")
        return None
    f_x = interp1d(cumulative_distances, x, kind='linear')
    f_y = interp1d(cumulative_distances, y, kind='linear')

    # 插值步长
    step = 0.1
    new_distances = np.arange(0, cumulative_distances[-1], step)

    # 插值结果
    new_x = f_x(new_distances)
    new_y = f_y(new_distances)

    # 合并插值结果为一个新的二维数组
    interpolated_points = np.column_stack((new_x, new_y))

    # 输出插值后的一维数组
    # result = interpolated_points.flatten()
    print(interpolated_points)
    return interpolated_points


def download_osm_data(bbox):
    print("bbox",bbox)

    # 提取并重新排序值
    north, south, east, west = bbox
    min_lon = west
    min_lat = south
    max_lon = east
    max_lat = north

    # 构造所需的 bbox 字符串
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    print(bbox)

    # bbox = "116.23,39.83,116.56,40.03"
    # print("bbox",bbox.type)
    url = f"http://overpass-api.de/api/map?bbox={bbox}"
    response = requests.get(url)
    with open('map.osm', 'wb') as file:
        file.write(response.content)
    osm_file_path = 'map.osm'
    G = ox.graph_from_xml(osm_file_path,simplify=False)

    return G

def osm_process(utm_coord,utm_to_info,directory):
    info = get_info_for_utm(utm_coord, utm_to_info)

    # 使用正则表达式提取数字部分
    match = re.search(r'\d+', info['filename'])
    if match:
        number_str = match.group()
        number = int(number_str)
        if number < 22 * 2:
            return None,None
        print("file_name",number)  # 输出: 22
    else:
        print("No number found in the filename.")
    

    # 定义文件路径和中心点
    center_point = (info['lat'], info['lon'])  # 替换为实际的经纬度值
    dist = 400 # 裁剪距离，单位米
    bbox = ox.utils_geo.bbox_from_point(center_point,dist=dist)

    try:
        print("come in ")
        graph_clipped = download_osm_data(bbox)
        print("come out")
        
        # # 尝试裁剪图以匹配边界框
        # graph_clipped = ox.truncate.truncate_graph_bbox(G, north, south, east, west, truncate_by_edge=True)
    except ValueError as e:
        # 捕获到 ValueError 时执行此代码块
        print(f"Warning: {e}. Proceeding with an empty graph.")
        # 如果需要，可以在这里定义一个空图或者采取其他措施
        graph_clipped = None  # 或者初始化一个新的空图 G_empty = ox.graph_from_point((lat, lon), dist=0)
        return None,None 


    # G = ox.graph.graph_from_point(center_point, dist=dist, simplify=False, truncate_by_edge=True)
    # graph_clipped = G
    # Directory containing the OXTS data files
    

    file_names = [f'{i:010}.txt' for i in range(number-22*2,number+78*2)]
    # Construct full file paths
    file_paths = [os.path.join(directory, name) for name in file_names]

    all_coords_data = read_oxts_files(file_paths)
    nodes = []
    # 获取all_coords_data的第一个和最后一个数据
    first_coord = all_coords_data[0] if all_coords_data else None
    last_coord = all_coords_data[-1] if len(all_coords_data) > 1 else first_coord

    # 遍历这两个数据项（如果存在）
    for coord in [c for c in (first_coord, last_coord) if c is not None]:
        node = ox.distance.nearest_nodes(graph_clipped, coord[0], coord[1])# 先lon,再lat
        nodes.append(node)

    try:
        route = nx.shortest_path(graph_clipped, nodes[0], nodes[-1], weight='length') 
    except nx.NetworkXNoPath:
        print(f"No path found between nodes[0] and nodes[-1].")
        return None,None
    print("nodes",nodes) ## 关键问题在这里，为什么route_utm为0/1个元素
    fig, ax = ox.plot_graph_route(graph_clipped, route) #可视化结果
    # 显示图形
    # plt.show()
    # 获取 nodes 和 edges 数据框
    nodes, edges = ox.graph_to_gdfs(graph_clipped)
    # 提取 route 中所有节点的经纬度信息
    route_nodes = nodes.loc[route]
    route_lat_lon = route_nodes[['y', 'x']]  # 'y' 是纬度，'x' 是经度

    # 将 route_lat_lon 中的经纬度转换为 UTM 坐标
    route_utm = route_lat_lon.apply(lambda row: gps_to_utm(row['y'], row['x']), axis=1)
    route_utm = pd.DataFrame(route_utm.tolist(), columns=['x', 'y'], index=route_lat_lon.index) 

    # Convert translations to a numpy array
    translations = np.array(route_utm)
    interpolated_points = cal_interp1d(translations)
    if interpolated_points is None:
        print("插值失败")
        return None,None
    target_point = utm_coord
    target_point = [target_point[0], target_point[1]]
    # 找到与 target_point 最近的点
    # print("target_point",target_point)
    # print("interpolated_points",interpolated_points)
    distances = np.linalg.norm(interpolated_points - target_point, axis=1)
    nearest_index = np.argmin(distances)
    current_point = interpolated_points[nearest_index]
    # print("current_point",current_point)

    # 计算每个点之间的距离
    distances_from_current = np.linalg.norm(interpolated_points - current_point, axis=1)
    # print(distances_from_current)

    # 创建一个轨迹数组
    trajectory_hmi = [current_point]


    # 找到后续每隔2米的点
    current_distance = 2
    count = 1
    max_iterations = 10000  # 设置最大迭代次数
    while count < 15 and max_iterations > 0:
        max_iterations -= 1
        for i, distance in enumerate(distances_from_current):
            if distance > current_distance and distance <= current_distance + 2 and i > nearest_index:
                trajectory_hmi.append(interpolated_points[i])
                current_distance += 2
                count += 1
                break

    # 将 trajectory_hmi 转换为 numpy 数组
    trajectory_hmi = np.array(trajectory_hmi)
    trajectory_hmi -= trajectory_hmi[0]
    trajectory_hmi = -trajectory_hmi

    # 输出结果
    print("trajectory_hmi",trajectory_hmi)

    # 创建一个轨迹数组
    trajectory_hmi_backward = []
    # 找到前面每隔2米的点
    current_distance = 0
    count = 0
    max_iterations = 10000  # 设置最大迭代次数
    while count < 5 and max_iterations > 0:
        max_iterations -= 1 # 防止无限循环
        for i, distance in enumerate(distances_from_current):
            if distance > current_distance and distance <= current_distance + 2 and i < nearest_index:
                trajectory_hmi_backward.append(interpolated_points[i] - current_point)
                current_distance += 2
                count += 1
                break

    # 将 trajectory_hmi_backward 转换为 numpy 数组
    trajectory_hmi_backward = -np.array(trajectory_hmi_backward)
    print("trajectory_hmi_backward",trajectory_hmi_backward)

    return trajectory_hmi,trajectory_hmi_backward
