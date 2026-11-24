# Publish FloodVision online (Render)

This project is configured for Render, but you must use your own hosting account and approve the deployment.

1. Create a GitHub account if you do not already have one, and create a new **private** repository.
2. Upload the extracted project files to that repository (do not upload private data or credentials).
3. Sign in to Render using your own account and choose **New > Blueprint**.
4. Connect the GitHub repository containing this project.
5. Render reads `render.yaml`; review the settings and create the service.
6. After the deployment succeeds, Render displays a public `https://...onrender.com` address.

Important:
- Do not share your password or verification codes with anyone.
- The website can be hosted before the AI model is trained, but prediction will show a clear message until `models/best_unet.pth` is created and included in the deployment.
- You need properly labelled flood images and masks, then run `python train.py` to create the checkpoint.
- Hosting providers can change their free-plan limits and may require a payment method. Check the current terms before publishing.
