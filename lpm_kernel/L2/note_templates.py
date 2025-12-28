"""Templates for formatting notes in different languages.

This module contains templates used to format subjective and objective notes in
different languages. These templates are used when generating training data from
user notes.
"""

from typing import Dict

SUBJECTIVE_TEMPLATES = {
    "English": {
        "basic": [
            "{user_name} recorded the following data:",
            "Here is the data recorded by {user_name}:",
            "The following is data recorded by {user_name}:"
        ],
        "title_suffix": "The topic of this data is: {title}. More specifically:"
    },
    "Chinese": {
        "basic": [
            "{user_name}记录了如下数据:",
            "{user_name}所记录的数据如下:", 
            "以下是{user_name}记录的数据:"
        ],
        "title_suffix": "该数据主题是:{title}。更具体的内容为："
    }
}

OBJECTIVE_TEMPLATES = {
    "English": {
        "with_content": [
            "Saved a third-party resource (link or file) to document {content}, which contains key points about {insight}.",
            "Archived an external resource to record {content}, with specific content about {insight}.",
            "Documented an external reference or link to preserve {content}, with discussions focusing on {insight}.",
            "Organized a third-party link or file for reference to {content}, with core topics being {insight}.",
            "Stored external material to record {content}, with main content covering {insight}.",
            "Backed up an external link or file to preserve information about {content}, with specific content being {insight}.",
            "Collected a third-party resource (link or file) to maintain records of {content}, with key focus on {insight}.",
            "Preserved an external file or link to document {content}, with main discussion around {insight}.",
            "Saved an external source link or file for future reference to {content}, with emphasis on {insight}.",
            "Archived external material intended to remember {content}, focusing on {insight}."
        ],
        "without_content": [
            "Saved a third-party resource (link or file) containing key points about {insight}.",
            "Archived an external resource with specific content about {insight}.",
            "Documented an external reference or link with discussions focusing on {insight}.",
            "Organized a third-party link or file with core topics being {insight}.",
            "Stored external material with main content covering {insight}.",
            "Backed up an external link or file with specific content being {insight}.",
            "Collected a third-party resource (link or file) with key focus on {insight}.",
            "Preserved an external file or link with main discussion around {insight}.",
            "Saved an external source link or file with emphasis on {insight}.",
            "Archived external material focusing on {insight}."
        ]
    },
    "Chinese": {
        "with_content": [
            "保存了一份第三方资料（链接或文件），旨在记录{content}，其中包含的要点是{insight}。",
            "归档了一个外部资源，其目的在于记下{content}，并涉及到的具体内容是{insight}。",
            "记录了一个外部文献或链接，目的是为了保存{content}，其相关讨论集中在{insight}。",
            "整理了一个第三方链接或文件，为了便于参考{content}，其核心主题为{insight}。",
            "存储了一份外部资料，意在记录{content}，并且该资料的主要内容涵盖了{insight}。",
            "备份了一个外部链接或文件，目的是为了保存关于{content}的信息，涉及的具体内容为{insight}。",
            "收录了一个第三方资源（链接或文件），主要为了保存{content}，讨论的重点是{insight}。",
            "保留了一份外部文件或链接，其主要目的是为了记录{content}，所讨论的内容主要围绕{insight}。",
            "保存了一个外部来源的链接或文件，为了日后参考{content}，其中的重点是{insight}。",
            "归档了外部资料，意图是为了记住{content}，并关注其中的{insight}。"
        ],
        "without_content": [
            "保存了一份第三方资料（链接或文件），其中包含的要点是{insight}。",
            "归档了一个外部资源，并涉及到的具体内容是{insight}。",
            "记录了一个外部文献或链接，其相关讨论集中在{insight}。",
            "整理了一个第三方链接或文件，其核心主题为{insight}。",
            "存储了一份外部资料，并且该资料的主要内容涵盖了{insight}。",
            "备份了一个外部链接或文件，涉及的具体内容为{insight}。",
            "收录了一个第三方资源（链接或文件），讨论的重点是{insight}。",
            "保留了一份外部文件或链接，所讨论的内容主要围绕{insight}。",
            "保存了一个外部来源的链接或文件其中的重点是{insight}。",
            "归档了外部资料，关注其中的{insight}。"
        ]
    }
}