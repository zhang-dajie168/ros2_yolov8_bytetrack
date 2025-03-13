from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_tracker',
            executable='object_tracker_node',
            name='object_tracker',
            parameters=['config/tracker_params.yaml'],
            output='screen',
            remappings=[
                # 如果 orbbec_camera 的话题名称不同，可在此重映射
                ('/camera/color/image_raw', '/camera/color/image_raw')
            ]
        )
    ])

