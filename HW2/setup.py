from setuptools import find_packages, setup

package_name = 'hw_2_solution'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/configs', ['configs/apriltags_position.yaml']),
        ('share/' + package_name + '/launch', ['launch/hw2_solution.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chenghao Li',
    maintainer_email='chl235@ucsd.edu',
    description='Homework 2 solution package',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motor_control = hw_2_solution.motor_control:main',
            'velocity_mapping = hw_2_solution.velocity_mapping:main',
            'hw2_solution = hw_2_solution.hw2_solution:main',
            'camera_tf = hw_2_solution.camera_tf:main',
        ],
    },
)
