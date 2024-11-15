import json
import cv2
import numpy as np
import os
import random
from datetime import datetime
from itertools import combinations

def image_process(cropped_image):
    # Chuyển đổi ảnh từ không gian màu BGR sang RGBA để thêm kênh alpha
    img_rgba = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA)
    
    # Chuyển đổi ảnh từ không gian màu BGR sang không gian màu HSV
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa khoảng màu trắng trong không gian màu HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    
    # Tạo mặt nạ cho các vùng có màu trắng
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Đảo ngược mặt nạ để giữ lại những phần không phải màu trắng
    mask_inv = cv2.bitwise_not(mask)
    
    # Thiết lập kênh alpha cho ảnh (vùng không trắng)
    img_rgba[:, :, 3] = mask_inv
    
    # Trả về ảnh đã được xử lý (nền trắng trong suốt)
    return img_rgba
###
# def paste_on_background(regions,
#                         background, 
#                         coords, 
#                         transcription, 
#                         gap_between_images=5):
#     """
#     Dán các vùng ảnh đã xử lý lên nền mới.
#     :param regions: Danh sách các vùng ảnh đã được cắt và xử lý
#     :param background: Ảnh nền nơi các vùng ảnh sẽ được dán lên
#     :param coords: Danh sách chứa vị trí các vùng để dánh ảnh
#     :param transcription: Danh sách transcription của từng vùng ảnh
#     :param gap_between_images: Khoảng cách giữa các ảnh
#     :return: Ảnh nền mới với các vùng ảnh được dán lên và danh sách label_coordinates
#     """
#     label_coordinates =[]# Danh sách lưu thông tin các label coordinates
#     for idx, img_region in enumerate(regions):
#         img_region_weights = []  # Danh sách trọng số chiều rộng của các ảnh con
#         new_image_regions = []  # Danh sách các vùng ảnh mới sau khi điều chỉnh
#         new_image_size = []  # Danh sách kích thước mới của các ảnh con
        
#         # Tính toán chiều rộng và chiều cao của vùng chỉ định
#         coord_idx_weight = coords[idx][2] - coords[idx][0]  # Chiều rộng vùng chỉ định
#         coord_idx_height = coords[idx][3] - coords[idx][1]  # Chiều cao vùng chỉ định
#         #print(coord_idx_height, coord_idx_weight)
#         # Tính toán trọng số chiều rộng của từng ảnh con
#         for img in img_region:
#             h, w = img.shape[:2]  # Lấy chiều cao và chiều rộng của ảnh
#             img_region_weights.append(w)

#         # Tính toán vị trí dán và kích thước mới của các ảnh con
#         start_x = 0
#         new_list_regions = []  # Danh sách các tọa độ vị trí dán ảnh con

#     #tinhs toan kich thuoc va vi tri moi cho cac vung anh
#     for w in img_region_weights:
#         #new_w =  round(((coord_idx_weight - gap_between_images * len(img_region_weights)) / sum(img_region_weights)) * w)
#         new_w = round((coord_idx_weight-(len(img_region_weights)-1)*gap_between_images)/len(img_region_weights))
#         new_image_size.append((new_w, coord_idx_height))
#         new_x1 = coords[idx][0] + start_x
#         new_x2 = new_x1 + new_w
#         new_y1 = coords[idx][1]
#         new_y2 = coords[idx][3]
#         new_list_regions.append((new_x1, new_x2, new_y1, new_y2))
#         print(new_list_regions)
#         start_x = new_x2 + gap_between_images
#     # dan cac anh con vao nen      
#     for i, img in enumerate(regions):
#         new_image = cv2.resize(img, new_image_size[i])
#         current_x, x_end, top_left_y, y_end = new_list_regions[i]

#         # h, w, _ = img.shape
#         # img_ratio = w / h  # Tính tỷ lệ chiều rộng và chiều cao của ảnh
#         # new_height = min(max_crop_height, h)  # Chiều cao tối đa cho ảnh
#         # new_width = int(new_height * img_ratio  )  # Tính chiều rộng tương ứng
#         # Tạo mặt nạ từ kênh alpha
#         alpha_channel = new_image[:, :, 3]  # Kênh alpha
#         mask = alpha_channel / 255.0  # Chuyển đổi kênh alpha sang định dạng [0, 1]

#         # Tính toán vùng cần thiết trên hình nền
#         # y_end = top_left_y + new_height
#         # x_end = current_x + new_width
#         background_region = background[top_left_y:y_end, current_x:x_end]

#         # Trộn ảnh đã xử lý vào hình nền dựa trên mặt nạ alpha
#         for c in range(3):  # Chỉ áp dụng cho 3 kênh màu (BGR)
#             background_region[:, :, c] = background_region[:, :, c] * (1 - mask) + new_image[:, :, c] * mask

#         # Cập nhật hình nền với vùng đã dán
#         background[top_left_y:y_end, current_x:x_end] = background_region

#         # Bổ sung thông tin transcription vào label_coordinates
#         label_coordinates.append({
#             'transcription': transcription[idx],  # Thêm thông tin về transcription
#             'points': [[current_x, top_left_y], 
#                        [x_end, top_left_y], 
#                        [x_end, y_end], 
#                        [current_x, y_end]],  # Các điểm tọa độ
#             'difficult': False  # Đánh dấu khó dễ (có thể tùy chỉnh)
#         })

#         # # Cập nhật vị trí dán cho ảnh tiếp theo
#         # current_x += new_width + gap_between_images

#     return background, label_coordinates  # Trả về hình nền mới và danh sách label coordinates
###
def paste_on_background(regions, 
                        background, 
                        coordinates,  # Đây là danh sách các tọa độ 
                        transcription, 
                        gap_between_images=5):
    """
    Dán các vùng ảnh đã xử lý lên nền mới.
    :param regions: Danh sách các vùng ảnh đã được cắt và xử lý
    :param background: Ảnh nền nơi các vùng ảnh sẽ được dán lên
    :param coordinates: Danh sách tọa độ chứa các vùng (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :param transcription: Danh sách transcription của từng vùng ảnh
    :param gap_between_images: Khoảng cách giữa các ảnh
    :return: Ảnh nền mới với các vùng ảnh được dán lên và danh sách label_coordinates
    """
    label_coordinates = []  # Danh sách lưu thông tin các label coordinates

    # Lặp qua từng tọa độ
    for  coord in coordinates:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = coord
        max_width = bottom_right_x - top_left_x
        max_height = bottom_right_y - top_left_y

        # Tính toán trọng số chiều rộng của các ảnh con và kích thước mới
        img_region_weights = [img.shape[1] for img in regions]  # Danh sách trọng số chiều rộng
        total_weight = sum(img_region_weights)  # Tổng chiều rộng của các ảnh con

        new_image_size = []
        start_x = top_left_x
        new_list_regions = []

        for w in img_region_weights:
            # Tính toán kích thước mới dựa trên kích thước vùng giới hạn
            new_w = round(((max_width - gap_between_images * len(img_region_weights)) / total_weight) * w)
            new_image_size.append((new_w, max_height))

            # Tính toán vị trí cho ảnh con
            new_x1 = start_x
            new_x2 = new_x1 + new_w
            new_y1 = top_left_y
            new_y2 = bottom_right_y
            new_list_regions.append((new_x1, new_x2, new_y1, new_y2))

            # Cập nhật vị trí bắt đầu cho ảnh con tiếp theo
            start_x = new_x2 + gap_between_images

        # Dán các ảnh con vào nền
        for index, img in enumerate(regions):
            new_image = cv2.resize(img, new_image_size[index])
            current_x, x_end, y1, y_end = new_list_regions[index]

            # Tạo mặt nạ từ kênh alpha
            alpha_channel = new_image[:, :, 3]
            mask = alpha_channel / 255.0

            # Tính toán vùng cần thiết trên hình nền
            background_region = background[y1:y_end, current_x:x_end]

            # Trộn ảnh đã xử lý vào hình nền dựa trên mặt nạ alpha
            for c in range(3):  # Chỉ áp dụng cho 3 kênh màu (BGR)
                background_region[:, :, c] = background_region[:, :, c] * (1 - mask) + new_image[:, :, c] * mask

            # Cập nhật hình nền với vùng đã dán
            background[y1:y_end, current_x:x_end] = background_region

            # Bổ sung thông tin transcription vào label_coordinates
            label_coordinates.append({
                'transcription': transcription[index],
                'points': [[current_x, y1], 
                           [x_end, y1], 
                           [x_end, y_end], 
                           [current_x, y_end]],
                'difficult': False
            })

    return background, label_coordinates


# Đọc file newlabel.txt
label_file = r"C:/Users/sythi/Downloads/HandWriting-copySaochep/HandWriting/0916_Data Samples 2/Label.txt"
error_image_folder = r"C:/Users/sythi/Downloads/HandWriting-copySaochep/HandWriting/0916_Data Samples 2/error_image1"
output_folder = r"C:/Users/sythi/Downloads/HandWriting-copySaochep/HandWriting/0916_Data Samples 2/newoutput27"
new_label_file_path = r"C:/Users/sythi/Downloads/HandWriting-copySaochep/HandWriting/0916_Data Samples 2/newnewlabel27.txt"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(error_image_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Đọc các dòng từ file newlabel.txt
with open(label_file, 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')  # Tách các dòng thành danh sách
    line_ids = list(range(len(lines[0:11])))
    subsets = list(combinations(line_ids, 5))
# Ghi vào file newnewlabel.txt
with open(new_label_file_path, 'a', encoding='utf-8') as new_label_file:
    batch_size = 5  # Lấy 5 ảnh mỗi lần
    img_batch_count = 1  # Đếm số batch ảnh
    current_date = datetime.now().strftime("%d%m")  # Lấy ngày hiện tại

    # Lặp qua từng batch ảnh # for batch_line in combination_line:
    for subset in subsets:         
        batch_lines = []  # Lấy 5 ảnh trong mỗi batch [lines[i] for i in batch]
        # In ra 5 dòng ảnh này
        for item in list(subset):
            batch_lines.append(lines[item])
        #print(batch_lines)
        regions = []  # Danh sách lưu các vùng ảnh đã xử lý
        list_contents = []  # Danh sách lưu transcription

        # Kiểm tra từng dòng để xử lý ảnh
        for line in batch_lines:
            if len(line) > 1:  # Kiểm tra dòng có đủ thông tin
                image_name, coordinates = line.split("	")  # Tách tên ảnh và tọa độ
                coordinates = json.loads(coordinates)  # Chuyển đổi tọa độ từ chuỗi JSON sang dạng Python
                image_path = os.path.join(r"C:/Users/sythi/Downloads/HandWriting-copySaochep/HandWriting", image_name)  # Tạo đường dẫn ảnh
                original_image = cv2.imread(image_path)  # Đọc ảnh gốc
                # Kiểm tra xem ảnh có được tải thành công không
                if original_image is None:
                    print(f"lỗi : không thể tải được ảnh {image_path}. vui lòng kiểm tra lại đường dẫn và định dạng ảnh")
                    continue
                # Khởi tạo danh sách để lưu trữ các phiên âm cho từng ảnh
                batch_transcriptions = []   
                #khởi tạo danh sách để lưu trữ vùng ảnh đã cắt của ảnh hiện tại
                image_regions = []
                # Xử lý từng tọa độ cắt trong ảnh
                for coord in coordinates:
                    x_tl, y_tl = coord['points'][0]  # Điểm trên cùng bên trái
                    x_br, y_br = coord['points'][2]  # Điểm dưới cùng bên phải

                    # Kiểm tra tọa độ cắt có hợp lệ không
                    if x_tl < 0 or y_tl < 0 or x_br > original_image.shape[1] or y_br > original_image.shape[0] or y_tl >= y_br:
                        print(f"Lỗi: Tọa độ cắt không hợp lệ cho ảnh {image_name}.")
                        # Lưu ảnh không hợp lệ vào thư mục error_image
                        error_image_path = os.path.join(error_image_folder, image_name)
                        cv2.imwrite(error_image_path, original_image)  # Lưu ảnh gốc không hợp lệ
                        continue
                    
                    # Kiểm tra chiều rộng và chiều cao của vùng cắt
                    if (x_br - x_tl) == 0 or (y_br - y_tl) == 0:
                        print(f"Lỗi: Vùng cắt có chiều rộng hoặc chiều cao bằng 0 cho ảnh {image_name}: {[x_tl, y_tl, x_br, y_br]}")
                        error_image_path = os.path.join(error_image_folder, image_name)
                        cv2.imwrite(error_image_path, original_image)  # Lưu ảnh gốc không hợp lệ
                        continue
                    
                    # Cắt vùng ảnh từ ảnh gốc theo tọa độ
                    cropped_image = original_image[y_tl:y_br, x_tl:x_br]
                    
                    # Kiểm tra xem ảnh cropped có hợp lệ không
                    if cropped_image.size == 0:
                        print(f"Lỗi: Vùng cắt không hợp lệ cho ảnh {image_name}.")
                        continue
                    
                    # Xử lý ảnh đã cắt
                    processed_image = image_process(cropped_image)
                    image_regions.append(processed_image)
                
                    #them phien am vao danh sach cho tung anh
                    batch_transcriptions.append(coord['transcription'])
                    
                #print(batch_transcriptions)    
                regions.append(image_regions)  # Thêm ảnh đã xử lý vào danh sách regions
                list_contents.append(batch_transcriptions)  # Thêm batch_transcription vào danh sách

        # Đọc ảnh nền từ file                           
        background = cv2.imread('D:/MyProject/load_json/light_cyan_background.png')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)  # Chuyển đổi sang không gian màu RGBA

        # # Định nghĩa các tọa độ cho các vùng ảnh lớn trên nền
        # coords = [
        #     (165, 355, 805, 380),
        #     (165, 400, 805, 425),
        #     (165, 445, 805, 470),
        #     (165, 490, 805, 515),
        #     (165, 535, 805, 560)
        # ]
        coords = [
            (80, 355, 960, 405),
            (80, 435, 960, 485),
            (80, 515, 960, 565),
            (80, 595, 960, 645),
            (80, 675, 960, 725)
        ]

        #chọn ngãu nhiên 1 chỉ số vùng để mở rộng
        random_index = random.randint(0,len(coords)-1)

        #mở rộng vùng được chọn ngẫu nhiên 
        expanded_coords = []
        for i, coord in enumerate(coords):
            if i == random_index:
                #tang kich thuoc vung len 
                x_tl, y_tl, x_br, y_br = coord
                expanded_coord = (x_tl + 40, y_tl - 30, x_br + 40, y_br + 30)# Điều chỉnh chiều cao và có thể cả chiều rộng nếu cần
                expanded_coords.append(expanded_coord)
            else: 
                expanded_coords.append(coord)
        #print(expanded_coords)            
        updated_label_coordinates = []  # Danh sách lưu thông tin label coordinates cho ảnh mới
        # Lặp qua từng tọa độ để dán ảnh vào nền
        for j, coord in enumerate(expanded_coords):
            print(expanded_coords)
            background, label_coordinates = paste_on_background(
                regions[j], background, [expanded_coords[j]], list_contents[j]
            )  # Dán ảnh và nhận label coordinates
            updated_label_coordinates.extend(label_coordinates)  # Cập nhật thông tin coordinates
        #print(updated_label_coordinates)
        # Tạo tên ảnh mới dựa trên ngày và số batch
        new_image_name = f"img{current_date}cyan_{img_batch_count}"
        img_batch_count += 1  # Tăng số batch lên 1

        # Lưu ảnh đã dán vào thư mục output
        output_path = os.path.join(output_folder, f"{new_image_name}.jpg")
        cv2.imwrite(output_path, background)

        # Ghi thông tin vào file newnewlabel.txt
        new_label_file.write(f"{new_image_name}" + "	" + f"{json.dumps(updated_label_coordinates, ensure_ascii=False)}\n")
        print(f"Tệp {new_label_file_path} đã được cập nhật.")
