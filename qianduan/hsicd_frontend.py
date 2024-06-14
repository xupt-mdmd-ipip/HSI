import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# 创建主窗口
root = tk.Tk()
root.title("HSICD 图像变化检测前端")
root.geometry("850x500")
root.configure(bg="#f0f0f0")

# 创建函数来选择图片
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        # 将图片缩放至合适的尺寸
        image.thumbnail((400, 400))
        # 创建Tkinter PhotoImage对象
        photo = ImageTk.PhotoImage(image)
        # 显示图片
        image_label.config(image=photo)
        image_label.image = photo  # 防止图片被垃圾回收

# 创建函数来显示预测的还原图
def display_prediction():
    prediction_image = Image.open('/change_map.png')
    prediction_image.thumbnail((400, 400))
    photo = ImageTk.PhotoImage(prediction_image)
    prediction_label.config(image=photo)
    prediction_label.image = photo

# 创建按钮和图片显示框架
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)

image_frame = tk.Frame(root, bg="#f0f0f0")
image_frame.pack(pady=10)

# 创建按钮来选择图片
choose_button = tk.Button(button_frame, text="选择图片", command=choose_image, padx=10, pady=5, bg="#4CAF50", fg="white", font=("Arial", 12))
choose_button.grid(row=0, column=0, padx=20)

# 创建按钮来显示预测的还原图
display_button = tk.Button(button_frame, text="显示预测的还原图", command=display_prediction, padx=10, pady=5, bg="#008CBA", fg="white", font=("Arial", 12))
display_button.grid(row=0, column=1, padx=20)

# 创建标签来显示选定的图片
image_label = tk.Label(image_frame, bg="#f0f0f0")
image_label.grid(row=0, column=0, padx=20)

# 创建标签来显示预测的还原图
prediction_label = tk.Label(image_frame, bg="#f0f0f0")
prediction_label.grid(row=0, column=1, padx=20)

# 运行主循环
root.mainloop()
