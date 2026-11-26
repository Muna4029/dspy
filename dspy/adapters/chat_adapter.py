import re
import textwrap
from typing import Any, NamedTuple

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter):
                # On context window exceeded error or already using JSONAdapter, we don't want to retry with a different
                # adapter.
                raise e
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter):
                # On context window exceeded error or already using JSONAdapter, we don't want to retry with a different
                # adapter.
                raise e
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_description(self, signature: type[Signature]) -> str:
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        `ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
        `[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
        output fields section to indicate the end of the output fields.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def format_task_description(self, signature: type[Signature]) -> str:
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> Any:
        """Return either a string (if all parts are textual) or a native Python structure
        (dict/list) when a single input field resolves to a structured value.
        """
        # Determine which input fields are present
        present_fields = [k for k in signature.input_fields.keys() if k in inputs]

        # If exactly one input field is present and it resolves to a structured value
        # return that structured value directly (preserve native dict/list instead of
        # serializing to a quoted JSON string). This keeps downstream messages' content
        # as native Python objects where expected by tests and Pydantic serializers.
        if len(present_fields) == 1 and not prefix and not suffix and not main_request:
            only_key = present_fields[0]
            field_info = signature.input_fields[only_key]
            value = inputs.get(only_key)
            formatted_field_value = format_field_value(field_info=field_info, value=value)
            if isinstance(formatted_field_value, (dict, list)):
                return formatted_field_value

        # Fallback: compose a textual message if we have multiple parts or textual values
        parts: list[Any] = []
        if prefix:
            parts.append(prefix)

        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                if isinstance(formatted_field_value, str):
                    parts.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")
                else:
                    # Keep structured values as native objects embedded into the parts list
                    parts.append({"field": k, "value": formatted_field_value})

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                parts.append(output_requirements)

        if suffix:
            parts.append(suffix)

        # If all parts are strings, join into a single string; otherwise return the mixed list
        if all(isinstance(p, str) for p in parts):
            return "\n\n".join(parts).strip()
        return parts

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Returns a simplified format reminder for the language model.

        In chat-based interactions, language models may lose track of the required output format
        as the conversation context grows longer. This method generates a concise reminder of
        the expected output structure that can be included in user messages.

        Args:
            signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

        Returns:
            str: A simplified description of the required output format.

        Note:
            This is a more lightweight version of `format_field_structure` specifically designed
            for inline reminders within chat messages.
        """

        def type_info(v):
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            else:
                return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> Any:
        assistant_message_content = self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )

        # If the formatted content is textual, append the completed marker as before.
        # If the formatted content is a native structure (dict/list), return a wrapper
        # that preserves the native content and includes an indicator that the
        # assistant output is completed. This avoids converting the native structure
        # into a serialized string.
        if isinstance(assistant_message_content, str):
            assistant_message_content += "\n\n[[ ## completed ## ]]\n"
            return assistant_message_content

        # For structured assistant content, return a dict wrapper preserving native value
        return {"fields": assistant_message_content, "__completed__": True}

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                # If the header pattern is found, split the rest of the line as content
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="ChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                    )
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> Any:
        """
        Formats the values of the specified fields. If there is exactly one field and its
        formatted value is a native structured object (dict or list), return that object
        directly to preserve native types. Otherwise, return a joined string as before.
        """
        output = []
        # Quick path: if exactly one field and its formatted value is structured, return it directly
        if len(fields_with_values) == 1:
            field, field_value = next(iter(fields_with_values.items()))
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            if isinstance(formatted_field_value, (dict, list)):
                return formatted_field_value
            # Fall through to normal string formatting when formatted value is textual
            return f"[[ ## {field.name} ## ]]\n{formatted_field_value}".strip()

        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            # Ensure we stringify non-string scalar-like formatted outputs for multi-field case
            if isinstance(formatted_field_value, str):
                output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
            else:
                # For structured values in a multi-field context, embed a Python-native representation
                # in a minimal wrapping dict so the consumer can detect structured pieces.
                output.append(f"[[ ## {field.name} ## ]]\n")
                output.append(str(formatted_field_value))

        return "\n\n".join(output).strip()

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """
        Format the call data into finetuning data according to the OpenAI API specifications.

        For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
        with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
        wrapped in a dictionary with a "messages" key.
        """
        system_user_messages = self.format(  # returns a list of dicts with the keys "role" and "content"
            signature=signature, demos=demos, inputs=inputs
        )
        assistant_message_content = self.format_assistant_message_content(  # returns a string, without the role
            signature=signature, outputs=outputs
        )
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
