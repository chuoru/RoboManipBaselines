import argparse
import io
import os

import cv2

# cv2's own __init__ points QT_QPA_PLATFORM_PLUGIN_PATH at its bundled
# cv2/qt/plugins dir (built against cv2's own bundled libxcb) so cv2.imshow
# works standalone. robo_manip_baselines/__init__.py separately preloads the
# system libxcb globally (to fix a PyAV/cv2.imshow deadlock -- see that
# file), which shadows cv2's bundled libxcb copy and makes cv2's bundled xcb
# plugin fail to load ("Could not load the Qt platform plugin xcb", aborting)
# -- this script doesn't use cv2.imshow, only matplotlib's Qt window
# (plt.show() below), so it's safe (and necessary) to drop the env var here
# and let Qt fall back to PyQt5's own bundled plugins dir instead, which is
# consistent with the preloaded system libxcb. Do NOT hoist this into
# robo_manip_baselines/__init__.py: teleop/TeleopBase.py's cv2.imshow camera
# preview needs QT_QPA_PLATFORM_PLUGIN_PATH left pointing at cv2's plugins.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    convert_depth_image_to_pointcloud,
    crop_and_resize,
    get_rot_pos_from_pose,
)


def plot_pose_triad(ax, position, rotation, length):
    """Draw one RGB xyz-axis triad (quiver arrows) at `position` oriented by
    `rotation`, on a 3D `Axes3D`. Makes rotation, not just position, visually
    inspectable along a plotted pose trajectory."""
    for axis_idx, color in enumerate(("r", "g", "b")):
        ax.quiver(*position, *(length * rotation[:, axis_idx]), color=color, linewidth=1.5)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str)
    parser.add_argument("--skip", default=10, type=int, help="skip", required=False)
    parser.add_argument(
        "-o",
        "--output_mp4_filename",
        type=str,
        required=False,
        help="save result as mp4 file when this option is set",
    )
    parser.add_argument(
        "--mp4_codec",
        type=str,
        default="mp4v",
    )
    parser.add_argument(
        "--rgb_crop_size_list",
        type=int,
        nargs="+",
        default=None,
        help="List of rgb image size (width, height) to be cropped before resize. Specify a 2-dimensional array if all images have the same size, or an array of <number-of-images> * 2 dimensions if the size differs for each individual image.",
    )
    parser.add_argument(
        "--use_pointcloud_camera",
        action="store_true",
        help="Whether to use the cameras specified by 'pointcloud_camera_names' instead of 'camera_names'",
    )
    parser.add_argument(
        "--display_stored_pointcloud",
        action="store_true",
        help="Whether to visualize the point cloud stored in the log file instead of generating it from depth image",
    )
    return parser.parse_args()


class VisualizeData:
    def __init__(
        self,
        rmb_filename,
        skip,
        output_mp4_filename,
        mp4_codec,
        rgb_crop_size_list,
        use_pointcloud_camera,
        display_stored_pointcloud,
    ):
        self.use_pointcloud_camera = use_pointcloud_camera
        if self.use_pointcloud_camera:
            self.display_stored_pointcloud = True
        else:
            self.display_stored_pointcloud = display_stored_pointcloud

        self.setup_data(rmb_filename, skip, rgb_crop_size_list)

        self.setup_plot()

        self.setup_video(output_mp4_filename, mp4_codec)

    def setup_data(self, rmb_filename, skip, rgb_crop_size_list):
        self.skip = skip

        # Use DataManager instead of RmbData because DataManager that puts all data in RAM is faster
        self.data_manager = DataManager(env=None)

        print(f"[{self.__class__.__name__}] Load {rmb_filename}")
        self.data_manager.load_data(rmb_filename)

        if self.use_pointcloud_camera:
            camera_names = self.data_manager.get_meta_data(
                "pointcloud_camera_names"
            ).tolist()
        else:
            camera_names = self.data_manager.get_meta_data("camera_names").tolist()

        try:
            rgb_tactile_names = self.data_manager.get_meta_data(
                "rgb_tactile_names"
            ).tolist()
        except KeyError:
            # To ensure backward compatibility
            rgb_tactile_names = self.data_manager.get_meta_data(
                "tactile_names"
            ).tolist()
        self.camera_names = camera_names + rgb_tactile_names

        if rgb_crop_size_list is None:
            self.rgb_crop_size_list = None
        else:
            # Set rgb image size list
            def refine_size_list(size_list):
                if len(size_list) == 2:
                    return [tuple(size_list)] * len(self.camera_names)
                else:
                    assert len(size_list) == len(self.camera_names) * 2
                    return [
                        (size_list[i], size_list[i + 1])
                        for i in range(0, len(size_list), 2)
                    ]

            self.rgb_crop_size_list = refine_size_list(rgb_crop_size_list)

        key_list = [
            DataKey.TIME,
            DataKey.COMMAND_JOINT_POS,
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.COMMAND_EEF_POSE,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
        ]
        self.data_list = {key: [] for key in key_list}

    def setup_plot(self):
        plt.rcParams["keymap.quit"] = ["q", "escape"]

        n_rows = len(self.camera_names) + 1
        n_cols = 5
        self.fig, self.ax = plt.subplots(
            n_rows,
            n_cols,
            figsize=(19.0, 12.0),
            constrained_layout=True,
            squeeze=False,
        )
        for ax_idx in range(1, n_rows):
            self.ax[ax_idx, 2].remove()
            self.ax[ax_idx, 3].remove()
            self.ax[ax_idx, 4].remove()
            self.ax[ax_idx, 2] = self.fig.add_subplot(
                n_rows, n_cols, n_cols * ax_idx + 3, projection="3d"
            )

        self.ax[0, 4].remove()
        self.ax[0, 4] = self.fig.add_subplot(n_rows, n_cols, n_cols, projection="3d")

        self.quit_flag = False
        self.scatter_list = [None] * len(self.camera_names)

        self.init_axes_limits()

    def init_axes_limits(self):
        time_range = (
            self.data_manager.get_data_seq(DataKey.TIME)[0],
            self.data_manager.get_data_seq(DataKey.TIME)[-1],
        )
        self.ax[0, 0].set_title("joint pos", fontsize=12)
        self.ax[0, 0].set_xlim(*time_range)
        self.ax[0, 1].set_title("joint vel", fontsize=12)
        self.ax[0, 1].set_xlim(*time_range)
        self.ax[0, 2].set_title("eef pose", fontsize=12)
        self.ax[0, 2].set_xlim(*time_range)
        self.ax[0, 3].set_title("eef wrench", fontsize=12)
        self.ax[0, 3].set_xlim(*time_range)
        for ax_idx, camera_name in enumerate(self.camera_names, start=1):
            self.ax[ax_idx, 0].set_title(f"{camera_name} rgb", fontsize=12)
            self.ax[ax_idx, 1].set_title(f"{camera_name} depth", fontsize=12)
            self.ax[ax_idx, 2].set_title(f"{camera_name} point cloud", fontsize=12)

        joint_pos = np.concatenate(
            [
                self.data_manager.get_data_seq(DataKey.COMMAND_JOINT_POS),
                self.data_manager.get_data_seq(DataKey.MEASURED_JOINT_POS),
            ]
        )
        self.ax[0, 0].set_ylim(joint_pos[:, :-1].min(), joint_pos[:, :-1].max())
        self.ax00_twin = self.ax[0, 0].twinx()
        self.ax00_twin.set_ylim(joint_pos[:, -1].min(), joint_pos[:, -1].max())

        joint_vel = self.data_manager.get_data_seq(DataKey.MEASURED_JOINT_VEL)
        self.ax[0, 1].set_ylim(joint_vel.min(), joint_vel.max())

        eef_pose = np.concatenate(
            [
                self.data_manager.get_data_seq(DataKey.MEASURED_EEF_POSE),
                self.data_manager.get_data_seq(DataKey.COMMAND_EEF_POSE),
            ]
        )
        self.ax[0, 2].set_ylim(eef_pose[:, 0:3].min(), eef_pose[:, 0:3].max())
        self.ax02_twin = self.ax[0, 2].twinx()
        self.ax02_twin.set_ylim(-1.0, 1.0)

        eef_pos = eef_pose[:, 0:3]
        self.pose3d_title = "eef pose trajectory (3D, RGB = xyz)"
        self.pose3d_xlim = (eef_pos[:, 0].min(), eef_pos[:, 0].max())
        self.pose3d_ylim = (eef_pos[:, 1].min(), eef_pos[:, 1].max())
        self.pose3d_zlim = (eef_pos[:, 2].min(), eef_pos[:, 2].max())
        self.pose3d_triad_length = 0.15 * max(np.ptp(eef_pos, axis=0).max(), 1e-3)
        self.setup_pose3d_axis()

        if DataKey.MEASURED_EEF_WRENCH in self.data_manager.all_data_seq.keys():
            eef_wrench = self.data_manager.get_data_seq(DataKey.MEASURED_EEF_WRENCH)
            self.ax[0, 3].set_ylim(eef_wrench.min(), eef_wrench.max())
        else:
            if self.ax[0, 3] in self.fig.axes:
                self.ax[0, 3].remove()

    def setup_pose3d_axis(self):
        # cla() (called each frame to clear the previous frame's trajectory
        # line/triads, since Axes3D.quiver's Line3DCollection artists aren't
        # picked up by the Line2D-only clear_axis() used elsewhere) wipes the
        # title/labels/limits too, so this re-applies them.
        ax = self.ax[0, 4]
        ax.set_title(self.pose3d_title, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(*self.pose3d_xlim)
        ax.set_ylim(*self.pose3d_ylim)
        ax.set_zlim(*self.pose3d_zlim)

    def setup_video(self, output_mp4_filename, mp4_codec):
        self.mp4_codec = mp4_codec
        self.output_mp4_filename = output_mp4_filename

        if self.output_mp4_filename:
            base, ext = os.path.splitext(self.output_mp4_filename)
            if ext.lower() != ".mp4":
                print(
                    f"[{self.__class__.__name__}] "
                    "Warning: "
                    f"The file '{self.output_mp4_filename}' has an incorrect extension '{ext}'. "
                    f"Changing it to '{base}.mp4'."
                )
                self.output_mp4_filename = base + ".mp4"

            output_mp4_dirname = os.path.dirname(self.output_mp4_filename)
            if output_mp4_dirname:
                os.makedirs(os.path.dirname(self.output_mp4_filename), exist_ok=True)
            width = int(self.fig.get_figwidth() * self.fig.dpi)
            height = int(self.fig.get_figheight() * self.fig.dpi)
            fourcc = cv2.VideoWriter_fourcc(*self.mp4_codec)
            self.video_writer = cv2.VideoWriter(
                self.output_mp4_filename, fourcc, 10, (width, height)
            )
        else:
            self.video_writer = None

    def handle_rgb_image(self, camera_idx, time_idx, rgb_key):
        ax_idx = camera_idx + 1
        self.ax[ax_idx, 0].axis("off")
        rgb_image = self.data_manager.get_single_data(rgb_key, time_idx)
        if self.rgb_crop_size_list is None:
            rgb_image_to_show = rgb_image
        else:
            rgb_image_to_show = crop_and_resize(
                rgb_image[np.newaxis], crop_size=self.rgb_crop_size_list[camera_idx]
            )[0]
        rgb_image_skip = 4
        self.ax[ax_idx, 0].imshow(rgb_image_to_show[::rgb_image_skip, ::rgb_image_skip])
        return rgb_image

    def handle_depth_image(self, camera_idx, time_idx, depth_key):
        ax_idx = camera_idx + 1
        if depth_key not in self.data_manager.all_data_seq.keys():
            if self.ax[ax_idx, 1] in self.fig.axes:
                self.ax[ax_idx, 1].remove()
            return None
        self.ax[ax_idx, 1].axis("off")
        depth_image = self.data_manager.get_single_data(depth_key, time_idx)
        depth_image_skip = 4
        self.ax[ax_idx, 1].imshow(depth_image[::depth_image_skip, ::depth_image_skip])
        return depth_image

    def handle_pointcloud(
        self,
        camera_idx,
        time_idx,
        camera_name,
        rgb_image,
        depth_image,
    ):
        ax_idx = camera_idx + 1
        if self.display_stored_pointcloud:
            pointcloud_key = DataKey.get_pointcloud_key(camera_name)
            if pointcloud_key not in self.data_manager.all_data_seq.keys():
                if self.ax[ax_idx, 2] in self.fig.axes:
                    self.ax[ax_idx, 2].remove()
                return
            xyzrgb_array = self.data_manager.get_data_seq(pointcloud_key)
            xyz_array = xyzrgb_array[time_idx, :, :3]
            rgb_array = xyzrgb_array[time_idx, :, 3:]
        else:
            depth_key = DataKey.get_depth_image_key(camera_name)
            if f"{depth_key}_fovy" not in self.data_manager.meta_data.keys():
                if self.ax[ax_idx, 2] in self.fig.axes:
                    self.ax[ax_idx, 2].remove()
                return
            pointcloud_skip = 10
            small_depth_image = depth_image[::pointcloud_skip, ::pointcloud_skip]
            small_rgb_image = rgb_image[::pointcloud_skip, ::pointcloud_skip]
            fovy = self.data_manager.get_meta_data(f"{depth_key}_fovy")
            xyz_array, rgb_array = convert_depth_image_to_pointcloud(
                small_depth_image,
                fovy=fovy,
                rgb_image=small_rgb_image,
                far_clip=3.0,  # [m]
            )
        if not xyz_array.size:
            return
        if self.scatter_list[ax_idx - 1] is None:

            def get_min_max(v_min, v_max):
                if self.display_stored_pointcloud:
                    return (v_min, v_max)
                else:
                    return (
                        0.75 * v_min + 0.25 * v_max,
                        0.25 * v_min + 0.75 * v_max,
                    )

            self.ax[ax_idx, 2].view_init(elev=-90, azim=-90)
            self.ax[ax_idx, 2].set_xlim(
                *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
            )
            self.ax[ax_idx, 2].set_ylim(
                *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
            )
            self.ax[ax_idx, 2].set_zlim(
                *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
            )
        else:
            self.scatter_list[ax_idx - 1].remove()
        self.ax[ax_idx, 2].axis("off")
        self.ax[ax_idx, 2].set_box_aspect(np.ptp(xyz_array, axis=0))
        self.scatter_list[ax_idx - 1] = self.ax[ax_idx, 2].scatter(
            xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
        )

    def plot(self):
        for time_idx in tqdm(
            range(0, len(self.data_manager.get_data_seq(DataKey.TIME)), self.skip),
            desc=self.ax[0, 0].plot.__name__,
        ):
            if self.quit_flag:
                break

            for key in self.data_list.keys():
                if key not in self.data_manager.all_data_seq.keys():
                    continue

                self.data_list[key].append(
                    self.data_manager.get_single_data(key, time_idx)
                )

            time_list = np.array(self.data_list[DataKey.TIME])

            self.clear_axis(self.ax[0, 0])
            self.clear_axis(self.ax00_twin)
            # TODO: It is assumed that the last joint has a different scale (e.g., gripper joint),
            # but this is not necessarily the case.
            self.ax[0, 0].plot(
                time_list,
                np.array(self.data_list[DataKey.COMMAND_JOINT_POS])[:, :-1],
                linestyle="--",
                linewidth=3,
            )
            self.ax[0, 0].set_prop_cycle(None)
            self.ax[0, 0].plot(
                time_list, np.array(self.data_list[DataKey.MEASURED_JOINT_POS])[:, :-1]
            )
            self.ax00_twin.plot(
                time_list,
                np.array(self.data_list[DataKey.COMMAND_JOINT_POS])[:, [-1]],
                linestyle="--",
                linewidth=3,
            )
            self.ax00_twin.set_prop_cycle(None)
            self.ax00_twin.plot(
                time_list, np.array(self.data_list[DataKey.MEASURED_JOINT_POS])[:, [-1]]
            )

            self.clear_axis(self.ax[0, 1])
            self.ax[0, 1].plot(
                time_list, np.array(self.data_list[DataKey.MEASURED_JOINT_VEL])[:, :-1]
            )

            self.clear_axis(self.ax[0, 2])
            self.clear_axis(self.ax02_twin)
            self.ax[0, 2].plot(
                time_list,
                np.array(self.data_list[DataKey.COMMAND_EEF_POSE])[:, :3],
                linestyle="--",
                linewidth=3,
            )
            self.ax[0, 2].set_prop_cycle(None)
            self.ax[0, 2].plot(
                time_list, np.array(self.data_list[DataKey.MEASURED_EEF_POSE])[:, :3]
            )
            self.ax02_twin.plot(
                time_list,
                np.array(self.data_list[DataKey.COMMAND_EEF_POSE])[:, 3:],
                linestyle="--",
                linewidth=3,
            )
            self.ax02_twin.set_prop_cycle(None)
            self.ax02_twin.plot(
                time_list, np.array(self.data_list[DataKey.MEASURED_EEF_POSE])[:, 3:]
            )

            if DataKey.MEASURED_EEF_WRENCH in self.data_manager.all_data_seq.keys():
                self.clear_axis(self.ax[0, 3])
                self.ax[0, 3].plot(
                    time_list, np.array(self.data_list[DataKey.MEASURED_EEF_WRENCH])
                )

            self.ax[0, 4].cla()
            self.setup_pose3d_axis()
            measured_pose_arr = np.array(self.data_list[DataKey.MEASURED_EEF_POSE])
            command_pose_arr = np.array(self.data_list[DataKey.COMMAND_EEF_POSE])
            self.ax[0, 4].plot3D(
                *measured_pose_arr[:, :3].T, color="tab:blue", label="measured"
            )
            self.ax[0, 4].plot3D(
                *command_pose_arr[:, :3].T,
                color="tab:orange",
                linestyle="--",
                label="command",
            )
            triad_stride = max(1, len(measured_pose_arr) // 15)
            for pose_idx in range(0, len(measured_pose_arr), triad_stride):
                rotation, position = get_rot_pos_from_pose(measured_pose_arr[pose_idx, :7])
                plot_pose_triad(
                    self.ax[0, 4], position, rotation, self.pose3d_triad_length
                )
            self.ax[0, 4].legend(fontsize=8)

            for camera_idx, camera_name in enumerate(self.camera_names):
                rgb_key = DataKey.get_rgb_image_key(camera_name)
                depth_key = DataKey.get_depth_image_key(camera_name)

                rgb_image = self.handle_rgb_image(camera_idx, time_idx, rgb_key)

                depth_image = self.handle_depth_image(camera_idx, time_idx, depth_key)

                self.handle_pointcloud(
                    camera_idx,
                    time_idx,
                    camera_name,
                    rgb_image,
                    depth_image,
                )

            plt.draw()
            plt.pause(0.001)

            if self.video_writer is not None:
                buf = io.BytesIO()
                self.fig.savefig(buf, format="jpg")
                buf.seek(0)
                img_array = np.frombuffer(buf.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.video_writer.write(img)
                buf.close()

            self.fig.canvas.mpl_connect("key_press_event", self.key_event)

        if self.video_writer is not None:
            self.video_writer.release()
            print(
                f"[{self.__class__.__name__}] "
                f"File '{self.output_mp4_filename}' has been successfully saved."
            )

        print(f"[{self.__class__.__name__}] Press 'Q' or 'Esc' to quit.")

        plt.show()

    def clear_axis(self, ax):
        for child in ax.get_children():
            if isinstance(child, plt.Line2D):
                child.remove()
        ax.set_prop_cycle(None)

    def key_event(self, event):
        if event.key in ["q", "escape"]:
            self.quit_flag = True


if __name__ == "__main__":
    args = parse_argument()
    viz = VisualizeData(**vars(parse_argument()))
    viz.plot()
