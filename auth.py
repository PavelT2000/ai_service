# import os
# from google.oauth2 import id_token
# from google.auth.transport import requests
# from fastapi import HTTPException, status, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# security = HTTPBearer()

# async def get_current_user(auth: HTTPAuthorizationCredentials = Depends(security)):
#     token = auth.credentials
#     try:
#         idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
#         return {
#             "email": idinfo['email'],
#             "name": idinfo.get('name')
#         }
#     except Exception:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Невалидный токен Google",
#         )