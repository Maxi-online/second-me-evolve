from pathlib import Path
import os
from typing import Union
from flask import send_from_directory, jsonify, redirect, Response
from .models import FileItem, DirectoryListing
from ...common.logging import logger


class FileServerHandler:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        logger.info(f"Initializing file server with base directory: {base_dir}")

    def handle_request(self, path: str, request_path: str) -> Union[Response, tuple]:
        """Handle file/directory access requests"""
        clean_path = path.rstrip("/")
        full_path = os.path.join(self.base_dir, clean_path)

        # security check
        try:
            Path(full_path).resolve().relative_to(Path(self.base_dir).resolve())
        except ValueError:
            return jsonify({"error": "Access denied"}), 403

        # check type and handle
        if os.path.isfile(full_path):
            return self._handle_file(clean_path, request_path)
        elif os.path.isdir(full_path):
            return self._handle_directory(clean_path, request_path)

        return jsonify({"error": "Not found"}), 404

    def _handle_file(
        self, clean_path: str, request_path: str
    ) -> Union[Response, tuple]:
        """Handle file access"""
        if request_path.endswith("/"):
            return redirect(f"/raw_content/{clean_path}", code=301)
        return send_from_directory(self.base_dir, clean_path, as_attachment=False)

    def _handle_directory(
        self, clean_path: str, request_path: str
    ) -> Union[Response, tuple]:
        """Handle directory access"""
        if not request_path.endswith("/"):
            return redirect(f"/raw_content/{clean_path}/", code=301)
        listing = self._list_directory(clean_path)
        return jsonify(
            listing.model_dump()
        )  # convert Pydantic model to dict, then to JSON response

    def _list_directory(self, path: str) -> DirectoryListing:
        """List directory content"""
        target_dir = os.path.join(self.base_dir, path)

        items = []
        for item in os.scandir(target_dir):
            item_type = "directory" if item.is_dir() else "file"
            item_size = os.path.getsize(item.path) if item.is_file() else None
            file_path = (Path(path) / item.name).as_posix()
            items.append(
                FileItem(
                    name=item.name,
                    type=item_type,
                    size=item_size,
                    path=file_path,
                    url=f"/raw_content/{file_path}"
                    if item.is_file()
                    else None,
                )
            )

        return DirectoryListing(
            current_path=path,
            items=sorted(items, key=lambda x: (x.type == "file", x.name)),
        )
