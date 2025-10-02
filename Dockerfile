# Dockerfile

# Start from the official OpenShift AI base image
FROM quay.io/opendatahub-contrib/workbench-images:jupyter-datascience-c9s-py311_2023c_latest

# Switch to the root user to install packages and fix permissions
USER root

# === Part 1: Install Python packages from requirements.txt ===
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# === Part 2: Copy and install our custom Python package ===
# Copy the entire package directory into a temporary location in the image
COPY ./cez_custom_package /tmp/cez_custom_package

# Install the package using pip.
RUN pip install /tmp/cez_custom_package && \
    rm -rf /tmp/cez_custom_package

# === Part 3: Fix permissions for OpenShift ===
# Grant the root group (gid=0) read/write permissions to the Python site-packages directory.
# This allows any user belonging to the root group to write cache files and other necessary things.
# The 'g+w' flag adds write permission for the group. The 'g+s' (setgid) ensures new files
# created in this directory inherit the group ID.
RUN chmod -R g+ws /opt/app-root/

# Also ensure the user's home directory is group-writable if needed.
# This path might vary depending on the base image, /opt/app-root/src is common.
RUN chmod -R g+ws /opt/app-root/src


# Switch back to the default notebook user.
# While OpenShift will override this UID, it's good practice for local testing.
USER 1001
