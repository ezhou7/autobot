from setuptools import setup, find_packages


def setup_package():
    metadata = dict(
        name="autobot",
        version="1.0.0",
        description="autobot",
        packages=find_packages(),
        package_data={
            "autobot": ["resources/*"]
        },
        include_package_data=True,
        entry_points={
            "console_scripts": [
                "autobot = autobot.main:main"
            ]
        },
        install_requires=[
            "numpy",
            "opencv-python",
            "torch",
            "ultralytics"
        ]
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
