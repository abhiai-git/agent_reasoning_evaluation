import os, json, boto3

class LLMProvider:
    def __init__(self, provider, model, api_key=None, aws_profile=None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.aws_profile = aws_profile
        self.client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            import openai
            openai.api_key = self.api_key or os.getenv("OPENAI_API_KEY") or input("OpenAI key: ")
            return openai

        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key or os.getenv("GEMINI_API_KEY") or input("Gemini key: "))
            return genai

        elif self.provider == "claude":
            from anthropic import Anthropic
            key = self.api_key or os.getenv("ANTHROPIC_API_KEY") or input("Claude key: ")
            return Anthropic(api_key=key)

        elif self.provider == "bedrock":
            profile = self.aws_profile or os.getenv("AWS_PROFILE") or input("AWS CLI profile: ")
            session = boto3.Session(profile_name=profile)
            return session.client("bedrock-runtime")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, prompt):
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

        elif self.provider == "gemini":
            model = self.client.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            return resp.text

        elif self.provider == "claude":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        elif self.provider == "bedrock":
            body = json.dumps({"inputText": prompt})
            resp = self.client.invoke_model(
                modelId=self.model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
            result = json.loads(resp["body"].read())
            return result.get("outputText", "")