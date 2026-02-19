from multi_generate_json_utils import *
import pickle

# Path to the OXTS data folder
folder_path = '/home/bdi/huihuixu/after_ddos_0228/trajectory-prediction/data/kitti/2011_09_30/2011_09_30_drive_0033_sync/oxts/data'

# Process all OXTS files in the folder starting from the 22nd file
utm_pose, utm_to_info = process_oxts_folder(folder_path)

data_utm_to_info = {
    'utm_to_info': utm_to_info
}
with open('data_utm_to_info09.pkl', 'wb') as file:
    print("come in data_utm_to_info")
    pickle.dump(data_utm_to_info, file)
    print("come out data_utm_to_info")

# # Example usage
# utm_coord = utm_pose[0]  # Example UTM coordinate
# info = get_info_for_utm(utm_coord, utm_to_info)
# if info:
#     print(f"UTM Coordinate: {utm_coord}")
#     print(f"Latitude: {info['lat']}, Longitude: {info['lon']}, Filename: {info['filename']}")
# else:
#     print("UTM coordinate not found.")

# Calculate the relative positions for each point in the UTM coordinates
trajectory_ins = calculate_relative_positions(utm_pose)

# # Optionally, print out the trajectory_ins dictionary
# for point, rel_positions in trajectory_ins.items():
#     print(f"Point: x={point[0]}, y={point[1]}, Zone Number={point[2]}, Zone Letter={point[3]}")
#     for dx, dy in rel_positions:
#         print(f"\tRelative Position: dx={dx}, dy={dy}")

# Calculate the past relative positions for each point in the UTM coordinates
trajectory_ins_past = calculate_past_relative_positions(utm_pose)

# Optionally, print out the trajectory_ins_past dictionary
# for point, rel_positions in trajectory_ins_past.items():
#     print(f"Point: x={point[0]}, y={point[1]}")
#     for dx, dy in rel_positions:
#         print(f"\tRelative Position: dx={dx}, dy={dy}")
trajectory_hmi_map = {}
trajectory_hmi_backward_map = {}
# 加载本地.osm文件
# 本地 .osm 文件路径
# osm_file_path = '/home/users/xzh/huihui_git/trajectory-prediction/karlsruhe_new.osm'
# G = ox.graph_from_xml(osm_file_path,simplify=False)
i = 0
save_interval = 10 # 每隔10次迭代保存一次数据
for utm_coord in utm_pose:
    i = i + 1
    print("第%d个轮次",i)
    if i < 20:
        continue
    # if i == 1466:
    #     break
    trajectory_hmi,trajectory_hmi_backward = osm_process(utm_coord,utm_to_info,folder_path)
    if (trajectory_hmi is not None and not trajectory_hmi.any()) or (trajectory_hmi_backward is not None and not trajectory_hmi_backward.any()):
    # 处理数组中没有任何真值的情况
        print("trajectory_hmi.any() == None or trajectory_hmi_backward.any() == None")
        continue
    key_utm_coord = tuple([utm_coord[0],utm_coord[1]])
    trajectory_hmi_map[key_utm_coord] = trajectory_hmi 
    trajectory_hmi_backward_map[key_utm_coord] = trajectory_hmi_backward  
        # 定期保存数据
    if i % save_interval == 0:
        data = {
            'trajectory_hmi_map': trajectory_hmi_map,
            'trajectory_hmi_backward_map': trajectory_hmi_backward_map,
            'utm_pose': utm_pose,
            'trajectory_ins': trajectory_ins,
            'trajectory_ins_past': trajectory_ins_past
        }
        with open('data09_%d.pkl' % i, 'wb') as file:
            print("data09_%d.pkl" % i)
            print("Saving checkpoint at iteration %d..." % i)
            pickle.dump(data, file)
            print("Checkpoint saved.")

# # 将所有数据结构放入一个字典中
# data = {
#     'trajectory_hmi_map': trajectory_hmi_map,
#     'trajectory_hmi_backward_map': trajectory_hmi_backward_map,
#     'utm_pose': utm_pose,
#     'trajectory_ins': trajectory_ins,
#     'trajectory_ins_past': trajectory_ins_past
# }
# # print("data",data)
# # 将数据写入 Pickle 文件
# with open('data0301.pkl', 'wb') as file:
#     print("come in")
#     pickle.dump(data, file)
#     print("come out")