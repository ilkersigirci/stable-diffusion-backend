from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from stable_diffusion_backend.db import sessions
from stable_diffusion_backend.db.models import Users
from stable_diffusion_backend.db.schemas import (
    auth as auth_schemas,
    users as user_schemas,
)
from stable_diffusion_backend.utils import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)

CurrentAsyncSession = Annotated[AsyncSession, Depends(sessions.get_async_session)]
CurrentFormData = Annotated[OAuth2PasswordRequestForm, Depends()]

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", summary="Register a new user")
async def register_user(
    payload: user_schemas.UsersCreate,
    db: CurrentAsyncSession,
) -> str:
    # querying database to check if user already exist
    q = await db.scalars(select(Users).filter(Users.email == payload.email))
    user = q.first()

    if user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exist",
        )

    user = Users(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        is_admin=payload.is_admin,
        hashed_password=get_password_hash(payload.password),
    )
    db.add(user)
    await db.commit()

    return "ok"


@router.post(
    "/login",
    summary="Create access and refresh tokens for user",
    response_model=auth_schemas.Token,
)
async def login(
    form_data: CurrentFormData,
    db: CurrentAsyncSession,
):
    # The name "username" must be used according to the OAuth spec but it containts
    # the users email address. Up to the frontend to implement this
    # See here: https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/#get-the-username-and-password
    q = await db.scalars(select(Users).filter(Users.email == form_data.username))
    user = q.first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )

    hashed_pass = user.hashed_password
    if not verify_password(form_data.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )

    jwt_data = {"sub": user.email}

    return {
        "access_token": create_access_token(jwt_data),
        "refresh_token": create_refresh_token(jwt_data),
    }
