"""
Module CBAM và SE: bản chính nằm ở src/adaptile/models/attention.py.

File này (trước ở utils/) giữ lại để code cũ trong legacy/ và checkpoint YOLO cũ
(pickle class dưới tên module `custom_modules`) vẫn import được.
"""

from adaptile.models.attention import CBAM, SE, ChannelAttention, SpatialAttention  # noqa: F401
