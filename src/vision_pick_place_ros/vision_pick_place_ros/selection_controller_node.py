#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SelectionController(Node):

    def __init__(self):
        super().__init__('selection_controller_node')

        self.publisher = self.create_publisher(
            String,
            '/vision/mission_command',
            10
        )

        self.subscription = self.create_subscription(
            String,
            '/vision/selected_object',
            self.selection_callback,
            10
        )

        self.get_logger().info(
            'Selection controller started. Waiting for A/B/C selection.'
        )

    def selection_callback(self, msg):
        value = msg.data.strip().upper()

        if value in ('A', 'B', 'C'):
            command = String()
            command.data = f'SELECT:{value}'
            self.publisher.publish(command)

            self.get_logger().info(
                f'Selected object {value} -> published {command.data}'
            )

        elif value.startswith('EXECUTE:'):
            target = value.split(':', 1)[1].strip()

            if target in ('A', 'B', 'C'):
                command = String()
                command.data = f'EXECUTE:{target}'
                self.publisher.publish(command)

                self.get_logger().info(
                    f'Execution requested for {target}'
                )

        elif value == 'RESET':
            command = String()
            command.data = 'RESET'
            self.publisher.publish(command)

            self.get_logger().info(
                'Reset command published.'
            )

        else:
            self.get_logger().warning(
                f'Unknown command received: {msg.data}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = SelectionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
