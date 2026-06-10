
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT=turing-course-437219-c0
ENV GOOGLE_CLOUD_LOCATION=us-central1

# Set up environment variables - End

# Install ADK - Start
RUN pip install google-adk==2.2.0
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/twin_ledger_internal/" "/app/agents/twin_ledger_internal/"

# Copy agent - End

# Install Agent Deps - Start
RUN pip install -r "/app/agents/twin_ledger_internal/requirements.txt"
# Install Agent Deps - End

EXPOSE 8080

CMD adk api_server --port=8080 --host=0.0.0.0 --session_service_uri=agentengine://projects/52245432644/locations/us-central1/reasoningEngines/9128662854017220608 --memory_service_uri=agentengine://projects/52245432644/locations/us-central1/reasoningEngines/9128662854017220608    --a2a  --gemini_enterprise_app_name=twin_ledger_internal "/app/agents"
