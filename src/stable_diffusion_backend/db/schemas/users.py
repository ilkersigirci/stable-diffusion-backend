from datetime import date

from pydantic import BaseModel, EmailStr, Field, constr

from stable_diffusion_backend.db.schemas.posts import Posts


class UsersBase(BaseModel):
    first_name: constr(to_lower=True)  # type: ignore
    last_name: constr(to_lower=True)  # type: ignore
    email: EmailStr
    is_admin: bool

    class Config:
        from_attributes = True


class UsersCreate(UsersBase):
    password: str = Field(alias="password")


class Users(UsersBase):
    id: int
    creation_date: date
    posts: list[Posts] = []
