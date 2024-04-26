import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.morphology import remove_small_holes


def make_output_file(triag_counts, triangle_centroids, filename="output.txt"):
    with open(filename, 'w') as f:
        f.write(f'{len(triangle_centroids)}\n')
        for i in range(len(triangle_centroids)):
            f.write(f'{triangle_centroids[i][0]}, {triangle_centroids[i][1]}; '
                    f'{triag_counts[i][0]}, {triag_counts[i][1]}, {triag_counts[i][2]}\n')
    messagebox.showinfo("Успех", f"Ответ был записан в файл {filename}")

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Segmentation App")

        # Создаем основной фрейм
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Создаем фрейм для отображения изображения (слева)
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Создаем фрейм для управления параметрами (справа)
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=50, pady=50)

        # Поле для отображения изображения
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Создаем элементы управления параметрами в правом фрейме
        self.create_control_elements()

        # Переменные
        self.image = None
        self.canny_image = None
        self.segmented_image = None
        self.image_with_points = None
        self.history = []

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.image = self.history[-1]
            self.display_image(self.image)
        elif len(self.history) == 1:
            self.display_image(self.history[0])

    def create_control_elements(self):
        self.load_button = tk.Button(self.control_frame, text="Загрузить изображение", command=self.load_image)
        self.load_button.pack(pady=10)

        self.undo_button = tk.Button(self.control_frame, text="Отменить последнее действие", command=self.undo)
        self.undo_button.pack(pady=10)

        self.exit_button = tk.Button(self.control_frame, text="Завершить программу", command=self.exit_app)
        self.exit_button.pack(pady=10)

        self.filter_type_label = tk.Label(self.control_frame, text="Выбрать фильтр")
        self.filter_type_label.pack(pady=10)

        self.filter_type_var = tk.StringVar()
        self.filter_type_combobox = ttk.Combobox(self.control_frame, textvariable=self.filter_type_var)
        self.filter_type_combobox['values'] = ('Двусторонний', 'Гауссовский', 'Медианный')
        self.filter_type_combobox.current(0)
        self.filter_type_combobox.pack(pady=0)

        self.d_label = tk.Label(self.control_frame, text="d (Двусторонний)")
        self.d_label.pack()

        self.d_entry = tk.Entry(self.control_frame)
        self.d_entry.insert(0, '20')
        self.d_entry.pack(pady=0)

        self.sigmaColor_label = tk.Label(self.control_frame, text="sigmaColor (Двусторонний)")
        self.sigmaColor_label.pack()

        self.sigmaColor_entry = tk.Entry(self.control_frame)
        self.sigmaColor_entry.insert(0, '30')
        self.sigmaColor_entry.pack(pady=0)

        self.sigmaSpace_label = tk.Label(self.control_frame, text="sigmaSpace (Двусторонний)")
        self.sigmaSpace_label.pack()

        self.sigmaSpace_entry = tk.Entry(self.control_frame)
        self.sigmaSpace_entry.insert(0, '100')
        self.sigmaSpace_entry.pack(pady=0)

        self.kernel_size_label = tk.Label(self.control_frame, text="Размер ядра")
        self.kernel_size_label.pack()

        self.kernel_size_entry = tk.Entry(self.control_frame)
        self.kernel_size_entry.insert(0, '3')
        self.kernel_size_entry.pack(pady=0)


        self.apply_filter_button = tk.Button(self.control_frame, text="Применить фильтр", command=self.apply_filter)
        self.apply_filter_button.pack(pady=10)

        self.binary_button = tk.Button(self.control_frame, text="Фильтр Кэнни", command=self.display_canny)
        self.binary_button.pack(pady=10)

        self.segment_button = tk.Button(self.control_frame, text="Найти фишки", command=self.display_trimino)
        self.segment_button.pack(pady=10)

        self.find_points_button = tk.Button(self.control_frame, text="Найти точки", command=self.display_points)
        self.find_points_button.pack(pady=10)

        self.output_button = tk.Button(self.control_frame, text="Классифицировать фишки", command=self.get_answer)
        self.output_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.history.append(self.image.copy())
            self.display_image(self.image)

    def display_image(self, image):
        # Создаем копию изображения для отображения
        display_image = image.copy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(display_image)
        image_tk = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def apply_filter(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        filter_type = self.filter_type_var.get()
        d = int(self.d_entry.get())
        sigmaColor = int(self.sigmaColor_entry.get())
        sigmaSpace = int(self.sigmaSpace_entry.get())
        kernel_size = int(self.kernel_size_entry.get())

        if filter_type == 'Гауссовский':
            filtered_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif filter_type == 'Двусторонний':
            filtered_image = cv2.bilateralFilter(self.image, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        elif filter_type == 'Медианный':
            filtered_image = cv2.medianBlur(self.image, kernel_size)
        else:
            filtered_image = self.image

        self.display_image(filtered_image)
        self.image = filtered_image
        self.history.append(self.image.copy())

    def canny(self):
        gray_image = self.image[:, :, 2]
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=350)
        return edges

    def display_canny(self):
        self.canny_image = self.canny()
        self.display_image(self.canny_image)
        self.history.append(self.canny_image.copy())

    def get_mask_trimino(self):
        edges = self.canny()
        edges = cv2.dilate(edges, np.ones((2, 2)))
        contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        edges = cv2.erode(edges, np.ones((7, 7)))
        edges = cv2.dilate(edges, np.ones((4, 4)))
        return edges

    def find_trimino(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        edges = self.get_mask_trimino()
        image_with_contours = self.history[-1].copy()
        triangles = []
        contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3:
                triangles.append(approx)

        cv2.drawContours(image_with_contours, triangles, -1, (0, 255, 0), 2)

        triangle_centroids = []

        for triangle in triangles:
            corners = triangle.reshape(3, 2)
            x_center = int(np.mean(corners[:, 0]))
            y_center = int(np.mean(corners[:, 1]))
            triangle_centroids.append((x_center, y_center))

        for triangle in triangles:
            corners = triangle.reshape(3, 2)
            for corner in corners:
                x, y = corner
                cv2.circle(image_with_contours, (x, y), 3, (0, 255, 0), -1)

        for i, center in enumerate(triangle_centroids):
            cv2.circle(image_with_contours, center, 3, (0, 255, 0), -1)

        return image_with_contours

    def display_trimino(self):
        image_with_contours = self.find_trimino()
        self.display_image(image_with_contours)
        self.segmented_image = image_with_contours
        self.history.append(self.segmented_image.copy())

    def find_points(self):
        edges = self.get_mask_trimino()
        mask = np.stack([edges // 255, edges // 255, edges // 255], axis=-1)
        masked_image = mask * self.image

        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        saturation = hsv_image[:, :, 1]
        smoothed_saturation = cv2.medianBlur(saturation, 3)
        new_edges = cv2.Canny(smoothed_saturation, threshold1=50, threshold2=350, apertureSize=5)

        new_edges = np.logical_xor(remove_small_holes(new_edges, area_threshold=300), new_edges)
        new_edges = np.where(new_edges == False, 0, 225).astype(np.uint8)
        new_edges = cv2.dilate(new_edges, np.ones((2, 2)))

        contours, _ = cv2.findContours(new_edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        c_arr = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                c_arr.append((cX, cY))
        return c_arr

    def display_points(self):
        image_with_contours = self.history[-1].copy()
        c_arr = self.find_points()
        for cX, cY in c_arr:
            cv2.circle(image_with_contours, (cX, cY), 2, (0, 0, 0), -1)
        self.display_image(image_with_contours)
        self.image_with_points = image_with_contours
        self.history.append(self.image_with_points.copy())

    def corners_points_and_centers(self):
        def euclidean_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        edges = self.get_mask_trimino()
        points = self.find_points()

        triag_counts = []
        n_triags, labels = cv2.connectedComponents(edges)
        n_triags -= 1
        triangle_centroids = []
        for i in range(n_triags):
            triag_mask = (labels == i + 1).astype('uint8')
            triag = edges * triag_mask

            contours, _ = cv2.findContours(triag, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 0.05 * cv2.arcLength(contours[0], True)
            triangle = cv2.approxPolyDP(contours[0], epsilon, True)
            if len(triangle) != 3:
                continue
            corners = triangle.reshape(3, 2)
            x_center = int(np.mean(corners[:, 0]))
            y_center = int(np.mean(corners[:, 1]))
            triangle_centroids.append((x_center, y_center))

            if len(triangle) != 3:
                continue

            corner_counts = [0, 0, 0]
            for point in points:
                if triag[point[1]][point[0]] != 0:
                    distances = []
                    for corner in triangle:
                        distances.append(euclidean_distance(point, corner[0]))
                    closest_corner_index = np.argmin(distances)
                    corner_counts[closest_corner_index] += 1
            triag_counts.append(corner_counts)
        return triag_counts, triangle_centroids

    def get_answer(self):
        triag_counts, triangle_centroids = self.corners_points_and_centers()
        make_output_file(triag_counts, triangle_centroids)

    def exit_app(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.attributes('-fullscreen', True)
    root.mainloop()

if __name__ == "__main__":
    main()
