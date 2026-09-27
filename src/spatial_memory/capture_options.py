"""Validated settings shared by capture, furniture review and texture generation."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_CAPTURE_PHOTOS = 64
MAX_CAPTURE_FRAMES = 120


FURNITURE_LABELS = ('bed', 'chair', 'sofa', 'desk', 'table', 'wardrobe', 'cabinet',
                    'bookshelf', 'curtain', 'coffee_table', 'dining_chair', 'locker', 'shelf', 'box')


class FurnitureOverride(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=120, pattern=r'^[\w.-]+$')
    label: str | None = None
    hidden: bool = False
    dimensions_m: tuple[float, float, float] | None = None

    @field_validator('label')
    @classmethod
    def valid_label(cls, value):
        if value is not None and value not in FURNITURE_LABELS:
            raise ValueError('対応していない家具の種類です')
        return value

    @field_validator('dimensions_m')
    @classmethod
    def valid_dimensions(cls, value):
        if value is not None and any(not .03 <= item <= 10 for item in value):
            raise ValueError('家具の寸法は0.03～10mで指定してください')
        return value


class CaptureOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    room_type: Literal['research', 'bedroom', 'storage', 'other'] | None = None
    up_axis: Literal['image-top', 'image-left', 'image-right', 'image-bottom'] = 'image-top'
    ceiling_height_m: float | None = Field(default=None, ge=1.5, le=10)
    reference_length_m: float | None = Field(default=None, ge=.3, le=50)
    reference_axis: Literal['longest_wall'] = 'longest_wall'
    furniture_frame_budget: int = Field(default=24, ge=8, le=32)
    furniture_overrides: list[FurnitureOverride] = Field(default_factory=list, max_length=80)
    texture_quality: Literal['standard', 'high'] = 'high'
    refine_cameras: bool = False
    max_frames: int = Field(default=60, ge=12, le=120)

    @model_validator(mode='after')
    def unique_overrides(self):
        ids = [item.id for item in self.furniture_overrides]
        if len(ids) != len(set(ids)):
            raise ValueError('同じ家具の修正が重複しています')
        return self
