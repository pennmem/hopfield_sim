from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from settings import settings
from model import HopfieldNetwork
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt
import time

import logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class App(QMainWindow):

    def __init__(self, config=settings):
        super(App, self).__init__()
        self.config = config
        self.setWindowTitle(config.window_title)

        self.setGeometry(0, 0, *self.config.window_size)

        model = HopfieldNetwork(config)
        model.add_directory(config.image_lib_dir)
        self.widget = HopfieldWidget(model, config, self)
        self.setCentralWidget(self.widget)
        self.show()


class HopfieldWidget(QWidget):

    def __init__(self, model, config, parent=None):
        super(QWidget, self).__init__(parent)
        self.model = model
        self.config = config
        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        self.animation = HopfieldAnimationWidget(model, config)
        self.settings = HopfieldSettingsWidget(model, config, self.animation)
        self.tabs.currentChanged.connect(self.settings.apply)

        self.tabs.addTab(self.animation, "Simulation")
        self.tabs.addTab(self.settings, "Settings")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

class HopfieldAnimationWidget(QWidget):

    def __init__(self, model, config):
        super(HopfieldAnimationWidget, self).__init__()
        self.going = False
        self.config = config

        self.layout = QHBoxLayout(self)

        side_bar = QVBoxLayout()

        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.go_pressed)
        self.go_button.setMaximumWidth(200)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.step_pressed)
        self.step_button.setMaximumWidth(200)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setMaximumWidth(200)


        self.iteration_label = QLabel("Iterations: 0")

        table_label = QLabel("Starting Image(s)")

        self.image_table = QTableWidget()
        self.image_table.setMinimumHeight(100)
        self.image_table.setRowCount(10)
        self.image_table.setColumnCount(2)
        self.image_table.setFixedWidth(150)
        self.image_table.verticalHeader().setVisible(False)
        self.image_table.horizontalHeader().setVisible(False)


        noise_box = QHBoxLayout()
        noise_label = QLabel("Percent Noise")
        noise_label.setMaximumWidth(100)
        self.noise_edit = QLineEdit("50")
        self.noise_edit.textChanged.connect(self.noise_edited)
        self.noise_edit.setMaximumWidth(100)
        noise_box.addWidget(noise_label)
        noise_box.addWidget(self.noise_edit)

        side_bar.addWidget(self.go_button)
        side_bar.addWidget(self.step_button)
        side_bar.addWidget(self.reset_button)

        side_bar.addWidget(self.iteration_label)
        side_bar.addWidget(table_label)

        side_bar.addWidget(self.image_table)
        side_bar.addLayout(noise_box)

        self.layout.addLayout(side_bar)

        self.figure = plt.figure()
        self.ax = self.figure.add_axes([.05,.05,.9,.9])
        self.ax.axis('off')

        self.canvas = FigureCanvas(self.figure)
        self.im = self.ax.imshow(np.random.choice([-1, 1], config.size, True), cmap='copper')
        self.figure.canvas.draw()

        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

        self.set_model(model)

        self.animation = None
        self.t_last = 0

    def go_pressed(self):
        if self.going:
            self.go_button.setText("Go")
            self.animation.event_source.stop()
        else:
            if self.animation is None:
                self.animation = FuncAnimation(self.figure, self.step, interval=.00, blit=False)
            self.go_button.setText("Stop")
            self.animation.event_source.start()
            self.figure.canvas.draw()
        self.going = not self.going

    def step_pressed(self):
        self.model.step()
        self.update_view()
        self.figure.canvas.draw()

    def reset(self):
        noise_percentage = float(self.noise_edit.text()) / 100
        log.debug("Noise percentage is {}".format(noise_percentage))
        ratios = self.get_initial_ratios()
        log.debug("Ratios are {}".format(ratios))
        self.model.reset_state(range(len(ratios)), ratios, noise_percentage)
        self.update_view()
        self.figure.canvas.draw()

    def step(self, *_):
        t1 = time.time()
        log.debug("Time since last {}".format(t1-self.t_last))
        self.model.step()
        t = time.time()
        log.debug("Step took {}".format(t-t1))
        self.update_view()
        log.debug("View update took {}".format(time.time()-t))
        self.t_last = time.time()

        if self.config.match_percent is not None and \
                self.model.state_matches(self.config.match_percent):
            self.animation.event_source.stop()
        return self.im,

    def size_changed(self):
        log.debug("Reset size to {}".format(self.config.size))
        self.im.remove()
        self.im = self.ax.imshow(np.random.choice([-1, 1], self.config.size, True),
                                 cmap='copper')

    def update_view(self):
        self.im.set_data(self.model.state_mat)
        self.iteration_label.setText("Iterations: {}".format(self.model.iterations))


    def get_initial_ratios(self):
        ratios = []
        for i in range(self.image_table.rowCount()):
            ratios.append(float(self.image_table.item(i, 1).text()))
        return ratios

    def set_model(self, model):
        self.model = model
        self.image_table.setRowCount(len(model.get_images()))
        for i, image in enumerate(model.get_images()):
            item = QTableWidgetItem(image.basename())
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.image_table.setItem(i, 0, item)
            self.image_table.setItem(i, 1, QTableWidgetItem("01" if i==0 else "00"))
        self.image_table.resizeRowsToContents()
        self.image_table.resizeColumnsToContents()


    def noise_edited(self):
        valid = True
        try:
            float(self.noise_edit.text())
        except:
            valid = False

        if not valid:
            self.noise_edit.setText("0")
        if self.noise_edit.text() != 'None' and \
                        float(self.noise_edit.text()) > 100:
            self.noise_edit.setText("100")


class HopfieldSettingsWidget(QWidget):

    @classmethod
    def LineEdit(cls, label):
        return QLineEdit(str(label))

    def __init__(self, model, config, anim_widget):
        super(HopfieldSettingsWidget, self).__init__()
        self.model = model
        self.config = config
        self.animation_widget = anim_widget

        layout = QVBoxLayout()

        self.lib_button = QPushButton("Image Lib Directory")
        self.lib_button.setMaximumWidth(200)
        self.lib_label = QLabel(config.image_lib_dir)

        self.lib_table = QTableWidget()
        self.lib_table.itemSelectionChanged.connect(self.image_selected)
        self.lib_table.cellChanged.connect(self.lib_strength_changed)
        self.lib_table.setFixedWidth(150)
        self.lib_table.setRowCount(10)
        self.lib_table.setColumnCount(2)
        self.lib_table.verticalHeader().setVisible(False)
        self.lib_table.horizontalHeader().setVisible(False)

        self.import_dir_button = QPushButton("Import Folder")
        self.import_image_button = QPushButton("Import Image")
        self.remove_image_button = QPushButton("Remove Image")

        speed_label = QLabel("Speed:")
        self.speed_edit = self.LineEdit(config.async_speed)
        self.speed_edit.textChanged.connect(self.speed_edited)

        size_label = QLabel("Image Size:")
        self.size_x_edit = self.LineEdit(config.size[0])
        self.size_x_edit.textChanged.connect(self.x_edited)
        self.size_y_edit = self.LineEdit(config.size[1])
        self.size_y_edit.textChanged.connect(self.y_edited)

        self.stop_on_match_cb = QCheckBox("Stop on match")
        perc_match_label = QLabel("% for match:")
        self.perc_match_edit = self.LineEdit(config.match_percent)
        self.perc_match_edit.textChanged.connect(self.perc_match_edited)

        self.temperature_cb = QCheckBox("Use Stochastic Dynamics")
        temperature_label = QLabel("Temperature:")
        self.temperature_edit = self.LineEdit(config.temperature)
        self.temperature_edit.textChanged.connect(self.temp_edited)

        self.flipped_rb = QRadioButton("Flipped")
        self.randomized_rb = QRadioButton("Randomized")

        self.synchronous_rb = QRadioButton("Synchronous")
        self.asynch_rb = QRadioButton("Asynchronous")

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)

        lib_layout = QHBoxLayout()
        lib_layout.addWidget(self.lib_button)
        lib_layout.addWidget(self.lib_label)
        layout.addLayout(lib_layout)

        sidebar = QVBoxLayout()

        table_layout = QHBoxLayout()
        table_layout.addWidget(self.lib_table)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.import_dir_button)
        button_layout.addWidget(self.import_image_button)
        button_layout.addWidget(self.remove_image_button)
        table_layout.addLayout(button_layout)
        sidebar.addLayout(table_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_edit)
        sidebar.addLayout(speed_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_x_edit)
        size_layout.addWidget(QLabel("x"))
        size_layout.addWidget(self.size_y_edit)
        sidebar.addLayout(size_layout)

        sidebar.addWidget(self.temperature_cb)
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(temperature_label)
        temp_layout.addWidget(self.temperature_edit)
        sidebar.addLayout(temp_layout)

        rb_layout = QHBoxLayout()

        noise_layout = QVBoxLayout()
        noise_group = QButtonGroup(self)
        noise_group.addButton(self.flipped_rb)
        noise_group.addButton(self.randomized_rb)
        noise_layout.addWidget(QLabel("Noise type:"))
        noise_layout.addWidget(self.flipped_rb)
        noise_layout.addWidget(self.randomized_rb)
        rb_layout.addLayout(noise_layout)

        update_layout = QVBoxLayout()
        update_group = QButtonGroup(self)
        update_group.addButton(self.synchronous_rb)
        update_group.addButton(self.asynch_rb)
        update_layout.addWidget(QLabel("Update type:"))
        update_layout.addWidget(self.synchronous_rb)
        update_layout.addWidget(self.asynch_rb)
        rb_layout.addLayout(update_layout)

        sidebar.addLayout(rb_layout)

        button_layout = QHBoxLayout()

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)

        sidebar.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(sidebar)

        self.figure = plt.figure()
        self.axes = self.figure.add_axes([.05,.05,.9,.9])
        self.canvas = FigureCanvas(self.figure)
        self.plot = self.axes.imshow(np.zeros(config.size), figure=self.figure, aspect='auto',
                                     cmap='bone')

        main_layout.addWidget(self.canvas)

        layout.addLayout(main_layout)

        self.setLayout(layout)

        self.set_model(model)
        self.set_config(config)

    def save(self):
        self.apply()
        filename,_ = QFileDialog.getSaveFileName(None, 'Save config', self.config.filename)
        if filename:
            self.config.save(str(filename))

    def load(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Load config', self.config.filename)
        if filename.text:
            self.config.load(filename.text)
            self.set_config(self.config)
            self.model.clear()
            self.model.add_directory(self.config.image_lib_dir)

    def lib_strength_changed(self):
        self.lib_table.resizeColumnsToContents()

        for i in range(self.lib_table.rowCount()):
            item = self.lib_table.item(i, 1)
            if item is None:
                continue
            strength = item.text()
            try:
                self.model.set_strength(i, float(strength))
            except Exception as e:
                log.error("Could not set strength to {}: {}".format(strength, e))

    def apply(self):
        self.config.async_speed = int(self.speed_edit.text())
        new_size = int(self.size_x_edit.text()), int(self.size_y_edit.text())
        size_changed = list(new_size) != list(self.config.size)
        if size_changed:
            log.debug("Size changed from {} to {}".format(self.config.size, new_size))
        self.config.size = new_size
        self.config.stop_percentage = float(self.perc_match_edit.text()) \
                if self.stop_on_match_cb.isChecked() else None
        self.config.temperature = float(self.temperature_edit.text()) \
                if self.temperature_cb.isChecked() else None
        self.config.flipped_noise = self.flipped_rb.isChecked()
        self.config.synchronous = self.synchronous_rb.isChecked()

        if size_changed:
            self.model.recalculate()
            self.animation_widget.size_changed()
            self.image_selected()

    def x_edited(self):
        if not self.size_x_edit.text().isdigit():
            self.size_x_edit.setText(str(self.config.size[0]))
        if int(self.size_x_edit.text()) > self.config.max_size:
            self.size_x_edit.setText(str(self.config.max_size))

    def y_edited(self):
        if not self.size_y_edit.text().isdigit():
            self.size_y_edit.setText(str(self.config.size[1]))
        if int(self.size_y_edit.text()) > self.config.max_size:
            self.size_y_edit.setText(str(self.config.max_size))

    def speed_edited(self):
        if not self.speed_edit.text().isdigit():
            self.speed_edit.setText(str(self.config.async_speed))

    def perc_match_edited(self):
        valid = True
        try:
            float(self.perc_match_edit.text())
        except:
            valid = False

        if not valid:
            self.perc_match_edit.setText(str(self.config.stop_percentage))
        if self.perc_match_edit.text() != 'None' and \
                        float(self.perc_match_edit.text()) > 100:
            self.perc_match_edit.setText("100")

    def temp_edited(self):
        if not self.temperature_edit.text().isdigit():
            self.temperature_edit.setText(str(self.config.temperature))

    def image_selected(self):
        selected = self.lib_table.currentIndex()
        image = self.model.get_image(selected.row())
        plt.imshow(image.matrix, figure=self.figure, cmap='bone')
        self.figure.canvas.draw()
        log.debug("Settings canvas updated")

    def set_model(self, model):
        self.model = model
        self.lib_table.setRowCount(len(model.get_images()))
        for i, image in enumerate(model.get_images()):
            strength = model.get_strength(i)
            item = QTableWidgetItem(image.basename())
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.lib_table.setItem(i, 0, item)
            self.lib_table.setItem(i, 1, QTableWidgetItem("%02d" % (strength)))
        self.lib_table.resizeColumnsToContents()
        self.lib_table.resizeRowsToContents()

    def set_config(self, config):
        self.config = config
        self.speed_edit.setText(str(config.async_speed))
        self.size_x_edit.setText(str(config.size[0]))
        self.size_y_edit.setText(str(config.size[1]))
        self.stop_on_match_cb.setChecked(config.stop_percentage is not None)
        self.perc_match_edit.setText(str(config.stop_percentage))
        self.temperature_cb.setChecked(config.temperature is not None)
        self.temperature_edit.setText(str(config.temperature))
        self.flipped_rb.setChecked(config.flipped_noise)
        self.randomized_rb.setChecked(not config.flipped_noise)
        self.synchronous_rb.setChecked(config.synchronous)
        self.asynch_rb.setChecked(not config.synchronous)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())