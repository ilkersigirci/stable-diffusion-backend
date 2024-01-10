from datetime import date

from pydantic import BaseModel


class PostsBase(BaseModel):
    title: str
    content: str

    class Config:
        from_attributes = True


class PostsCreate(PostsBase):
    user_id: int


class Posts(PostsBase):
    id: int
    creation_date: date
    user_id: int
