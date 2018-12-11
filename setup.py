from setuptools import setup

setup(name='drl_course',
      url='https://github.com/yoproject/drl_course',
      author='SH & OK',
      packages=['acrobot', 'taxi_dqn', 'taxi_pg'],
      install_requires=['gym', 'torch', 'torchvision'],
      zip_safe=False)