from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import logging

import numpy as np


DEFAULT_EMBEDDING_DIM = 1536
DISTANCE_RATE = 0.8


class TimeType(str, Enum):
    RECENT = "recent"
    EARLIER = "earlier"


MIN_MEMORIES_N = {TimeType.RECENT: 3, TimeType.EARLIER: 10}

TIME_RANGE = {TimeType.RECENT: 60 * 60 * 24 * 1, TimeType.EARLIER: 60 * 60 * 24 * 7}


class MemoryType(str, Enum):
    TEXT = "TEXT"
    MARKDOWN = "MARKDOWN"
    PDF = "PDF"
    LINK = "LINK"


class AnalysisType(str, Enum):
    SUBJECT = "SUBJECT"
    OBJECT = "OBJECT"
    CHAT = "CHAT"


def datetime2timestamp(time_str: str) -> float:
    """Convert datetime string to timestamp.
    
    Args:
        time_str: String representation of datetime in TIME_FORMAT format.
        
    Returns:
        Timestamp in seconds.
        
    Raises:
        Exception: If time_str has invalid format.
    """
    try:
        timestamp = datetime.strptime(time_str, TIME_FORMAT).timestamp()
        return timestamp
    except Exception as e:
        logging.error(f"Invalid time format: {time_str}")
        raise e


OBJECT_NOTE_TYPE = [MemoryType.LINK]
SUBJECT_NOTE_TYPE = [
    MemoryType.TEXT,
    MemoryType.MARKDOWN,
    MemoryType.PDF,
]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


TAG_TYPE = {
    TimeType.RECENT: {"time": "Today", "default": "Recent"},
    TimeType.EARLIER: {"time": "Earlier", "default": "Earlier"},
}


class Chunk:
    """Represents a chunk of document content with embedding information."""
    
    def __init__(
        self,
        id: int,
        document_id: int,
        content: str,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        tags: Optional[List[str]] = None,
        topic: Optional[str] = None,
    ):
        """Initialize a Chunk instance.
        
        Args:
            id: Unique identifier for the chunk.
            document_id: ID of the document this chunk belongs to.
            content: Text content of the chunk.
            embedding: Vector representation of the chunk content.
            tags: List of tags associated with the chunk.
            topic: Topic classification for the chunk.
        """
        self.id = id
        self.document_id = document_id
        self.content = content
        self.embedding = embedding.squeeze() if embedding is not None else None
        self.tags = tags
        self.topic = topic


class Note:
    """Represents a note with its content and metadata."""
    
    def __init__(
        self,
        noteId: int = None,
        content: str = "",
        createTime: str = "",
        memoryType: str = "",
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        chunks: List[Chunk] = None,
        title: str = "",
        summary: str = "",
        insight: str = "",
        tags: List[str] = None,
        topic: str = None,
    ):
        """Initialize a Note instance.
        
        Args:
            noteId: Unique identifier for the note.
            content: Text content of the note.
            createTime: Creation timestamp in string format.
            memoryType: Type of the memory (TEXT, MARKDOWN, etc.).
            embedding: Vector representation of the note content.
            chunks: List of chunks the note is divided into.
            title: Title of the note.
            summary: Summary of the note content.
            insight: Insights extracted from the note.
            tags: List of tags associated with the note.
            topic: Topic classification for the note.
        """
        self.id = noteId
        self.content = content
        self.create_time = createTime
        self.memory_type = memoryType
        self.embedding = embedding.squeeze() if embedding is not None else None
        self.chunks = chunks or []
        self.title = title
        self.summary = summary
        self.insight = insight
        self.tags = tags
        self.topic = topic

    def __str__(self) -> str:
        """Return a string representation of the note.
        
        Returns:
            Formatted string with note metadata and content.
        """
        note_statement = "---\n"
        if self.id:
            note_statement += f"[ID]: {self.id}\n"
        if self.title:
            note_statement += f"[Title]: {self.title}\n"
        if self.create_time:
            note_statement += f"[Date]: {self.create_time}\n"
        if self.memory_type:
            note_statement += f"[Type]: {self.memory_type}\n"
        note_statement += "---\n\n"
        if self.summary:
            note_statement += f"----- Doc Summary -----\n{self.summary}\n\n"
        if self.insight:
            note_statement += f"----- Doc Insight -----\n{self.insight}\n\n"
        if not (self.insight or self.summary):
            note_statement += f"----- Doc Content -----\n{self.content[:4000]}\n\n"

        return note_statement

    def to_json(self) -> Dict[str, Any]:
        """Convert the note to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the note.
        """
        if hasattr(self, "processed"):
            return {
                "id": self.id,
                "insight": self.insight,
                "summary": self.summary,
                "memory_type": self.memory_type,
                "create_time": self.create_time,
                "title": self.title,
                "content": self.content,
                "processed": self.processed,
            }
        else:
            return {
                "id": self.id,
                "insight": self.insight,
                "summary": self.summary,
                "memory_type": self.memory_type,
                "create_time": self.create_time,
                "title": self.title,
                "content": self.content,
            }

    def to_str(self, analysis_type: AnalysisType = None) -> str:
        """Convert the note to a string based on analysis type.
        
        Args:
            analysis_type: Type of analysis to determine format.
            
        Returns:
            Formatted string representation of the note.
            
        Raises:
            ValueError: If memory_type or analysis_type is invalid.
        """
        if not analysis_type:
            if self.memory_type in SUBJECT_NOTE_TYPE:
                analysis_type = AnalysisType.SUBJECT
            elif self.memory_type in OBJECT_NOTE_TYPE:
                analysis_type = AnalysisType.OBJECT
            else:
                raise ValueError(f"Invalid memory type: {self.memory_type}")

        if analysis_type == AnalysisType.SUBJECT:
            return self.to_subject_str()
        elif analysis_type == AnalysisType.OBJECT:
            return self.to_object_str()
        else:
            raise ValueError(f"Invalid analysis type: {analysis_type}")

    def to_subject_str(self) -> str:
        """Convert the note to a string formatted as subject.
        
        Returns:
            Formatted string for subject analysis.
        """
        note_statement = "---\n"
        if self.id:
            note_statement += f"[ID]: {self.id}\n"
        if self.title:
            note_statement += f"[Title]: {self.title}\n"
        if self.create_time:
            note_statement += f"[Date]: {self.create_time}\n"
        if self.memory_type:
            note_statement += f"[Type]: {self.memory_type}\n"
        note_statement += "---\n\n"
        if self.summary:
            note_statement += f"----- Doc Summary -----\n{self.summary}\n\n"
        if self.insight:
            note_statement += f"----- Doc Insight -----\n{self.insight}\n\n"
        if not (self.insight or self.summary):
            note_statement += f"----- Doc Content -----\n{self.content[:4000]}\n\n"

        return note_statement

    def to_object_str(self) -> str:
        """Convert the note to a string formatted as object.
        
        Returns:
            Formatted string for object analysis.
        """
        note_statement = "---\n"
        if self.id:
            note_statement += f"[ID]: {self.id}\n"
        if self.title:
            note_statement += f"[Title]: {self.title}\n"
        if self.create_time:
            note_statement += f"[Read Time]: {self.create_time}\n"
        if self.memory_type:
            note_statement += f"[Meta Type]: {self.memory_type}\n"
        note_statement += "---\n\n"
        if self.summary:
            note_statement += f"----- Doc Summary -----\n{self.summary}\n\n"
        if not self.summary and self.insight:
            note_statement += f"----- Doc Insight -----\n{self.insight}\n\n"
        if not (self.insight or self.summary):
            note_statement += f"----- Doc Content -----\n{self.content[:4000]}\n\n"
        return note_statement


class Memory:
    def __init__(self, memoryId: int, embedding: List[float] = None):
        self.memory_id = memoryId

        if embedding is not None:
            self.embedding = np.array(embedding).squeeze()
        else:
            self.embedding = None

    def to_json(self):
        return {"memoryId": self.memory_id}


class Cluster:
    def __init__(
        self,
        clusterId: int,
        memoryList: List[Optional[Union[Dict, Memory]]] = [],
        centerEmbedding: List[float] = None,
        is_new=False,
    ):
        self.cluster_id = clusterId
        memory_list = [
            memory if isinstance(memory, Memory) else Memory(**memory)
            for memory in memoryList
        ]
        self.memory_list = memory_list
        self.is_new = is_new
        self.size = len(memory_list)
        self.cluster_center = (
            np.array(centerEmbedding)
            if centerEmbedding
            else np.zeros(DEFAULT_EMBEDDING_DIM)
        )
        self.merge_list = []

    def add_memory(self, memory: Memory):
        self.memory_list.append(memory)
        self.size += 1
        self.get_cluster_center()

    def extend_memory_list(self, memory_list: List[Memory]):
        self.memory_list.extend(memory_list)
        self.size += len(memory_list)
        self.get_cluster_center()

    def get_cluster_center(self):
        if not self.memory_list:
            self.cluster_center = np.zeros(DEFAULT_EMBEDDING_DIM)
        else:
            self.cluster_center = np.mean(
                [memory.embedding for memory in self.memory_list], axis=0
            )

    def prune_outliers_from_cluster(self):
        if not self.memory_list:
            self.get_cluster_center()
        memory_list = sorted(
            self.memory_list,
            key=lambda x: np.linalg.norm(x.embedding - self.cluster_center),
        )
        memory_list = memory_list[: max(int(self.size * DISTANCE_RATE), 1)]
        self.memory_list = memory_list
        self.size = len(memory_list)
        self.get_cluster_center()

    def to_json(self):
        return {
            "clusterId": self.cluster_id if not self.is_new else None,
            "memoryList": [memory.to_json() for memory in self.memory_list],
            "centerEmbedding": self.cluster_center.tolist(),
            "mergeList": self.merge_list,
        }


class ShadeTimeline:
    def __init__(
        self,
        refMemoryId: int = None,
        createTime: str = "",
        descSecondView: str = "",
        descThirdView: str = "",
        is_new: bool = False,
    ):
        self.create_time = createTime
        self.ref_memory_id = refMemoryId
        self.desc_second_view = descSecondView
        self.desc_third_view = descThirdView
        self.is_new = is_new

    @classmethod
    def from_raw_format(cls, raw_format: Dict[str, Any]):
        return cls(
            refMemoryId=raw_format.get("refMemoryId", None),
            createTime=raw_format.get("createTime", ""),
            descSecondView="",
            descThirdView=raw_format.get("description", ""),
            is_new=True,
        )

    def add_second_view(self, description):
        self.desc_second_view = description

    def to_json(self):
        return {
            "createTime": self.create_time,
            "refMemoryId": self.ref_memory_id,
            "descThirdView": self.desc_third_view,
            "descSecondView": self.desc_second_view,
        }


class ConfidenceLevel(str, Enum):
    VERY_LOW = "VERY LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"


CONFIDENCE_LEVELS_INT = {
    ConfidenceLevel.VERY_LOW: 1,
    ConfidenceLevel.LOW: 2,
    ConfidenceLevel.MEDIUM: 3,
    ConfidenceLevel.HIGH: 4,
    ConfidenceLevel.VERY_HIGH: 5,
}


class ShadeInfo:
    def __init__(
        self,
        id: int = None,
        name: str = "",
        aspect: str = "",
        icon: str = "",
        descThirdView: str = "",
        contentThirdView: str = "",
        descSecondView: str = "",
        contentSecondView: str = "",
        timelines: List[Dict[str, Any]] = [],
        confidenceLevel: str = None,
    ):
        self.id = id
        self.name = name
        self.aspect = aspect
        self.icon = icon
        self.desc_second_view = descSecondView
        self.desc_third_view = descThirdView
        self.content_third_view = contentThirdView
        self.content_second_view = contentSecondView
        if confidenceLevel:
            self.confidence_level = ConfidenceLevel(confidenceLevel)
        else:
            self.confidence_level = None

        self.timelines = [ShadeTimeline(**timeline) for timeline in timelines]

    def imporve_shade_info(
        self,
        improveDesc: str,
        improveContent: str,
        improveTimelines: List[Dict[str, Any]],
    ):
        self.desc_third_view = improveDesc
        self.content_third_view = improveContent
        self.timelines.extend(
            [ShadeTimeline.from_raw_format(timeline) for timeline in improveTimelines]
        )

    def add_second_view(
        self,
        domainDesc: str,
        domainContent: str,
        domainTimeline: List[Dict[str, Any]],
        *args,
        **kwargs,
    ):
        self.desc_second_view = domainDesc
        self.content_second_view = domainContent
        timelime_dict = {
            timelime.ref_memory_id: timelime for timelime in self.timelines
        }
        for timeline in domainTimeline:
            ref_memory_id = timeline.get("refMemoryId", None)
            if not (ref_memory_id and ref_memory_id in timelime_dict):
                logging.error(
                    f"Timeline with refMemoryId {ref_memory_id} already exists, skipping"
                )
                continue
            timelime_dict[ref_memory_id].add_second_view(
                timeline.get("description", "")
            )

    def _preview_(self, second_view: bool = False):
        if second_view:
            return f"- **{self.name}**: {self.desc_second_view}"
        return f"- **{self.name}**: {self.desc_third_view}"

    def to_str(self):
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.desc_third_view}\n\n**[Content]**: \n{self.content_third_view}\n"
        shade_statement += "---\n\n[Timelines]:\n"
        for timeline in self.timelines:
            shade_statement += f"- {timeline.create_time}, {timeline.desc_third_view}, {timeline.ref_memory_id}\n"
        return shade_statement

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descSecondView": self.desc_second_view,
            "descThirdView": self.desc_third_view,
            "contentThirdView": self.content_third_view,
            "contentSecondView": self.content_second_view,
            "confidenceLevel": self.confidence_level if self.confidence_level else None,
            "timelines": [timeline.to_json() for timeline in self.timelines],
        }


class AttributeInfo:
    def __init__(
        self,
        id: int = None,
        name: str = "",
        description: str = "",
        confidenceLevel: Optional[Union[str, ConfidenceLevel]] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        if confidenceLevel and isinstance(confidenceLevel, str):
            self.confidence_level = ConfidenceLevel(confidenceLevel)
        elif isinstance(confidenceLevel, ConfidenceLevel):
            self.confidence_level = confidenceLevel
        else:
            self.confidence_level = None

    def to_str(self):
        # - **[Attribute Name]**: (Attribute Description), Confidence level: [LOW/MEDIUM/HIGH]
        return f"- **{self.name}**: {self.description}, Confidence level: {self.confidence_level.value}"

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "confidenceLevel": self.confidence_level.value
            if self.confidence_level
            else None,
        }


class Bio:
    def __init__(
        self,
        contentThirdView: str = "",
        content: str = "",
        summaryThirdView: str = "",
        summary: str = "",
        attributeList: List[Dict[str, Any]] = [],
        shadesList: List[Dict[str, Any]] = [],
    ):
        self.content_third_view = contentThirdView
        self.content_second_view = content
        self.summary_third_view = summaryThirdView
        self.summary_second_view = summary
        self.attribute_list = sorted(
            [AttributeInfo(**attribute) for attribute in attributeList],
            key=lambda x: CONFIDENCE_LEVELS_INT[x.confidence_level],
            reverse=True,
        )
        self.shades_list = sorted(
            [ShadeInfo(**shade) for shade in shadesList],
            key=lambda x: len(x.timelines),
            reverse=True,
        )

    def to_str(self) -> str:
        global_bio_statement = ""
        if self.is_raw_bio():
            global_bio_statement += (
                f"**[Origin Analysis]**\n{self.summary_third_view}\n"
            )
        # global_bio_statement += f"**[Identity Attributes]**\n"
        # global_bio_statement += '\n'.join([attribute.to_str() for attribute in self.attribute_list])

        global_bio_statement += f"\n**[Current Shades]**\n"
        for shade in self.shades_list:
            global_bio_statement += shade.to_str()
            global_bio_statement += "\n==============\n"
        return global_bio_statement

    def complete_content(self, second_view: bool = False) -> str:
        interests_preference_field = (
            "\n### User's Interests and Preferences ###\n"
            + "\n".join([shade._preview_(second_view) for shade in self.shades_list])
        )
        if not second_view:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_third_view
        else:
            conclusion_field = "\n### Conclusion ###\n" + self.summary_second_view
        return f"""## Comprehensive Analysis Report ##
{interests_preference_field}
{conclusion_field}"""

    def is_raw_bio(self) -> bool:
        if not self.content_third_view and not self.summary_third_view:
            return True
        return False

    def to_json(self) -> Dict[str, Any]:
        return {
            "contentThirdView": self.content_third_view,
            "content": self.content_second_view,
            "summaryThirdView": self.summary_third_view,
            "summary": self.summary_second_view,
            "shadesList": [shade.to_json() for shade in self.shades_list],
        }


class ShadeMergeInfo:
    def __init__(
        self,
        id: int = None,
        name: str = "",
        aspect: str = "",
        icon: str = "",
        desc_third_view: str = "",
        content_third_view: str = "",
        desc_second_view: str = "",
        content_second_view: str = "",
        cluster_info: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.name = name
        self.aspect = aspect
        self.icon = icon
        self.desc_second_view = desc_second_view
        self.desc_third_view = desc_third_view
        self.content_third_view = content_third_view
        self.content_second_view = content_second_view
        self.cluster_info = cluster_info

    def improve_shade_info(self, improveDesc: str, improveContent: str):
        self.desc_third_view = improveDesc
        self.content_third_view = improveContent

    def add_second_view(self, domainDesc: str, domainContent: str):
        self.desc_second_view = domainDesc
        self.content_second_view = domainContent

    def _preview_(self, second_view: bool = False):
        if second_view:
            return f"- **{self.name}**: {self.desc_second_view}"
        return f"- **{self.name}**: {self.desc_third_view}"

    def to_str(self):
        shade_statement = f"---\n**[Name]**: {self.name}\n**[Aspect]**: {self.aspect}\n**[Icon]**: {self.icon}\n"
        shade_statement += f"**[Description]**: \n{self.desc_third_view}\n\n**[Content]**: \n{self.content_third_view}\n"
        shade_statement += "---\n\n"
        if self.cluster_info:
            shade_statement += (
                f"**[Cluster Info]**: \n{json.dumps(self.cluster_info, indent=2)}\n"
            )
        return shade_statement

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "aspect": self.aspect,
            "icon": self.icon,
            "descSecondView": self.desc_second_view,
            "descThirdView": self.desc_third_view,
            "contentThirdView": self.content_third_view,
            "contentSecondView": self.content_second_view,
            "clusterInfo": self.cluster_info,
        }


class ShadeMergeResponse:
    def __init__(self, result: Any, success: bool):
        self.success: bool = success
        self.message: str = ""
        self.merge_shade_list: Optional[List[Dict[str, Any]]] = None

        if not success:
            self.message = result if isinstance(result, str) else "Error occurred"
            logging.error(self.message)
        else:
            self.message = "Success"
            self.merge_shade_list = result.get("mergeShadeList")

    def to_json(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "mergeShadeList": self.merge_shade_list,
        }


class Todo:
    def __init__(
        self,
        todoId: int = 0,
        content: str = "",
        deadlineTime: str = "",
        createTime: str = "",
        status: str = "Done",
    ) -> None:
        self.todo_id = todoId
        self.content = content
        self.deadline_time = deadlineTime
        self.create_time = createTime
        self.status = status

    def __str__(self):
        todo_statement = "---\n"
        todo_statement += f"[Action] User have a Plan\n"
        if self.content:
            todo_statement += f"[Content]: {self.content}\n"
        if self.create_time:
            todo_statement += f"[Create Time]: {self.create_time}\n"
        if self.deadline_time:
            todo_statement += f"[Deadline Time]: {self.deadline_time}\n"
        if self.status:
            todo_statement += f"[Status]: {self.status}\n"
        return todo_statement


class Chat:
    def __init__(
        self,
        sessionId: str = "",
        summary: str = "",
        title: str = "",
        createTime: str = "",
    ) -> None:
        self.session_id = sessionId
        self.summary = summary
        self.title = title
        self.create_time = createTime

    def __str__(self):
        chat_statement = "---\n"
        chat_statement += f"[Action] User had a chat\n"
        if self.create_time:
            chat_statement += f"[Create Time]: {self.create_time}\n"
        if self.title:
            chat_statement += f"[Title]: {self.title}\n"
        if self.summary:
            chat_statement += f"{self.summary}\n"
        return chat_statement


class UserInfo:
    def __init__(
        self, cur_time: str, notes: List[Note], todos: List[Todo], chats: List[Chat]
    ):
        self.notes = notes
        self.todos = todos
        self.chats = chats
        self.cur_time = cur_time
        self.recent_tag = {k: v["default"] for k, v in TAG_TYPE.items()}
        self.memories = sorted(
            notes + todos + chats,
            key=lambda x: datetime2timestamp(x.create_time),
            reverse=True,
        )
        self.recent_memories = self.get_range_memories(TimeType.RECENT)
        self.earlier_memories = self.get_range_memories(TimeType.EARLIER)[
            len(self.recent_memories) :
        ]

    def __str__(self):
        user_memories_statement = "### {recent_type} Memory ###\n".format(
            recent_type=self.recent_tag[TimeType.RECENT]
        )
        user_memories_statement += "".join(
            [str(memory) for memory in self.recent_memories]
        )
        user_memories_statement += "\n\n### Earlier Memory ###\n"
        user_memories_statement += "".join(
            [str(memory) for memory in self.earlier_memories]
        )
        return user_memories_statement

    def get_range_memories(self, time_type: TimeType) -> List[Union[Note, Todo, Chat]]:
        if len(self.memories) < MIN_MEMORIES_N[time_type]:
            return self.memories
        recent_memories = []
        cur_datetime = datetime.fromtimestamp(datetime2timestamp(self.cur_time))
        end_datetime = cur_datetime + timedelta(days=1)
        end_timestamp = end_datetime.replace(hour=0, minute=0, second=0).timestamp()
        for memory in self.memories:
            if (
                end_timestamp - datetime2timestamp(memory.create_time)
                < TIME_RANGE[time_type]
            ):
                recent_memories.append(memory)
            else:
                break
        if len(recent_memories) >= MIN_MEMORIES_N[time_type]:
            self.recent_tag[time_type] = TAG_TYPE[time_type].get(
                "time", TAG_TYPE[time_type].get("default")
            )
            return recent_memories
        return self.memories[: MIN_MEMORIES_N[time_type]]
