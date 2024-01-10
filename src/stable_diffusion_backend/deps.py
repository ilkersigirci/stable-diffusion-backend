from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from jose.exceptions import JWTError
from pydantic import ValidationError
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

from .utils import ALGORITHM, JWT_SECRET_KEY

reuseable_oauth = OAuth2PasswordBearer(tokenUrl="/auth/login", scheme_name="JWT")

CurrentAsyncSession = Annotated[AsyncSession, Depends(sessions.get_async_session)]
CurrentToken = Annotated[str, Depends(reuseable_oauth)]


async def get_current_user(
    token: CurrentToken,
    db: CurrentAsyncSession,
) -> user_schemas.Users:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        token_data = auth_schemas.TokenPayload(**payload)

        if (
            not token_data.exp
            or datetime.fromtimestamp(token_data.exp) < datetime.now()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (JWTError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    q = await db.scalars(select(Users).filter(Users.email == token_data.sub))
    user = q.first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )

    return user
