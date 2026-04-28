import os
import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# 1. Carga automática del .env
load_dotenv() 

fastapi_app = FastAPI()

SYSTEM_PROMPT = """
Eres un asistente de edicion de workflows para un sistema BPM.
Tu tarea NO es opinar: tu tarea es EDITAR el workflow y devolver una propuesta ejecutable por backend.

REGLA CRITICA DE SALIDA:
- Responde UNICAMENTE con JSON valido.
- No agregues markdown.
- No agregues explicaciones.
- No agregues comentarios.
- No agregues texto antes o despues del JSON.

OBJETIVO:
- Recibiras un workflow actual + una solicitud de cambio del usuario.
- Debes devolver el workflow completo actualizado.
- Mantener todos los campos que no fueron solicitados para cambio.
- Aplicar cambios minimos y consistentes.

CONTRATO DE DATOS (workflow):
- id: string (ObjectId hex) opcional si no existe aun
- codigo: string
- nombre: string
- descripcion: string
- estado: "borrador" | "activo" | "inactivo"
- enEdicionPor: string|null (ObjectId hex)
- createdBy: string|null (ObjectId hex)
- createdAt: string|null (ISO-8601)
- updatedAt: string|null (ISO-8601)
- lanes: Lane[]
- nodes: WorkflowNode[]
- edges: WorkflowEdge[]

Lane:
- id: string (obligatorio)
- nombre: string
- descripcion: string
- responsable: string
- color: string
- departmentId: string (ObjectId hex, obligatorio)
- orden: number

WorkflowNode:
- id: string (obligatorio y unico)
- tipo: "inicio" | "actividad" | "decision" | "paralelo_inicio" | "paralelo_fin" | "fin"
- nombre: string
- descripcion: string
- laneId: string (debe existir en lanes[].id)
- departmentId: string (ObjectId hex, debe coincidir con el departmentId de su lane)
- responsableTipo: "cliente" | "usuario" | "departamento" (obligatorio para nodo actividad)
- responsableUsuarioId: string|null (obligatorio si responsableTipo="usuario")
- responsableRole: "Cliente" | "Funcionario" | "Disenador" | "Administrador" | null
- slaMinutos: number|null
- permiteAdjuntos: boolean
- form: NodeForm|null
- decisionRule: DecisionRule|null
- posicionX: number
- posicionY: number

NodeForm:
- titulo: string
- descripcion: string
- campos: FormField[]

FormField:
- id: string (obligatorio y unico dentro del nodo)
- label: string (obligatorio)
- tipo: "text" | "textarea" | "number" | "date" | "bool" | "select" | "file"
- required: boolean
- options: string[]
- placeholder: string

DecisionRule:
- field: string
- operator: "EQUALS" | "NOT_EQUALS" | "GREATER_THAN" | "GREATER_EQUAL_THAN" | "LESS_THAN" | "LESS_EQUAL_THAN" | "CONTAINS" | "STARTS_WITH" | "ENDS_WITH"
- value: string
- onTrueDestinoNodeId: string
- onFalseDestinoNodeId: string

WorkflowEdge:
- fromNodeId: string (debe existir)
- toNodeId: string (debe existir)
- tipo: "secuencial" | "iterativo" | "paralelo"
- label: string|null

REGLAS ESTRUCTURALES OBLIGATORIAS (backend):
1. Debe existir al menos un nodo tipo "inicio".
2. Debe existir al menos un nodo tipo "fin".
3. Si hay nodos, debe haber lanes.
4. lane.id unico, y cada lane debe tener departmentId.
5. Un mismo departmentId no puede repetirse en mas de una lane.
6. Cada node.laneId debe existir en lanes.
7. node.departmentId debe ser igual al departmentId de su lane.
8. node.id unico y no vacio.
9. Cada edge debe apuntar a nodos existentes.
10. Nodo "decision" debe tener decisionRule.
11. Nodo "paralelo_inicio" debe tener >= 2 aristas salientes.
12. Nodo "paralelo_fin" debe tener >= 2 aristas entrantes.
13. Nodo "actividad" debe tener responsableTipo.
14. Si actividad.responsableTipo == "usuario", responsableUsuarioId es obligatorio.
15. Si actividad.responsableTipo == "departamento", departmentId obligatorio.
16. Si un nodo tiene form.campos, cada campo debe tener id, label y tipo.

SEMANTICA DE EJECUCION IMPORTANTE:
- El motor soporta join implicito: si un nodo de tipo "actividad" tiene mas de 1 arista entrante, se activa solo cuando llegaron TODAS las ramas.
- Aristas tipo "iterativo" representan retorno de bucle.
- En decisiones se evalua decisionRule usando contexto de formularios completados.

REGLAS DE EDICION:
- Prioriza siempre la instruccion del usuario sobre cualquier regla generica de preservacion.
- No elimines nodos, lanes ni edges por defecto.
- Si el usuario pide eliminar, reducir, simplificar o recortar el workflow, entonces si puedes eliminar nodos, lanes o edges que correspondan a esa solicitud.
- No inventes ObjectId. Usa SOLO ids permitidos en catalogos de entrada.
- Si falta informacion obligatoria para cumplir solicitud, conserva estructura valida y elige alternativa segura.
- No elimines nodos/lanes/edges no solicitados, excepto que el usuario lo pida.
- Mantener estabilidad de ids existentes para no romper referencias.

VALIDACION PREVIA A RESPUESTA (obligatorio):
- Verifica que TODOS los ids referenciados existan.
- Verifica que laneId y departmentId de cada nodo sean consistentes.
- Verifica que decisionRule apunte a nodos existentes.
- Verifica que el JSON final sea parseable.

FORMATO FINAL DE RESPUESTA:
- Debe ser un unico objeto JSON con el workflow completo actualizado.
""".strip()


class DecisionRule(BaseModel):
    field: str | None = None
    operator: Literal[
        "EQUALS",
        "NOT_EQUALS",
        "GREATER_THAN",
        "GREATER_EQUAL_THAN",
        "LESS_THAN",
        "LESS_EQUAL_THAN",
        "CONTAINS",
        "STARTS_WITH",
        "ENDS_WITH",
    ] | None = None
    value: str | None = None
    onTrueDestinoNodeId: str
    onFalseDestinoNodeId: str


class FormField(BaseModel):
    id: str
    label: str
    tipo: Literal["text", "textarea", "number", "date", "bool", "select", "file"]
    required: bool = False
    options: list[str] = Field(default_factory=list)
    placeholder: str | None = None


class NodeForm(BaseModel):
    titulo: str | None = ""
    descripcion: str | None = ""
    campos: list[FormField] = Field(default_factory=list)


class WorkflowNode(BaseModel):
    id: str
    tipo: Literal["inicio", "actividad", "decision", "paralelo_inicio", "paralelo_fin", "fin"]
    nombre: str
    descripcion: str | None = ""
    laneId: str
    departmentId: str
    responsableTipo: Literal["cliente", "usuario", "departamento"] | None = None
    responsableUsuarioId: str | None = None
    responsableRole: Literal["Cliente", "Funcionario", "Disenador", "Administrador"] | None = None
    slaMinutos: int | None = None
    permiteAdjuntos: bool = False
    form: NodeForm | None = None
    decisionRule: DecisionRule | None = None
    posicionX: float = 0
    posicionY: float = 0


class WorkflowLane(BaseModel):
    id: str
    nombre: str | None = ""
    descripcion: str | None = ""
    responsable: str | None = ""
    color: str | None = ""
    departmentId: str
    orden: int = 0


class WorkflowEdge(BaseModel):
    fromNodeId: str
    toNodeId: str
    tipo: Literal["secuencial", "iterativo", "paralelo"] = "secuencial"
    label: str | None = None


class WorkflowModel(BaseModel):
    id: str | None = None
    codigo: str = ""
    nombre: str = ""
    descripcion: str | None = ""
    estado: Literal["borrador", "activo", "inactivo"] = "borrador"
    enEdicionPor: str | None = None
    createdBy: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None
    lanes: list[WorkflowLane] = Field(default_factory=list)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)


class DepartmentCatalogItem(BaseModel):
    id: str
    nombre: str


class UserCatalogItem(BaseModel):
    id: str
    nombre: str
    departmentId: str


class CatalogsPayload(BaseModel):
    departments: list[DepartmentCatalogItem] = Field(default_factory=list)
    users: list[UserCatalogItem] = Field(default_factory=list)
    allowedResponsableRoles: list[str] = Field(default_factory=list)


class RulesPayload(BaseModel):
    mustReturnOnlyJson: bool = True
    preserveExistingIds: bool = True
    strictValidation: bool = True


class EditWorkflowRequest(BaseModel):
    requestId: str | None = None
    mode: Literal["edit_workflow"] = "edit_workflow"
    userInstruction: str
    currentWorkflow: WorkflowModel
    catalogs: CatalogsPayload
    rules: RulesPayload = Field(default_factory=RulesPayload)


def _validate_workflow_structure(workflow: WorkflowModel, catalogs: CatalogsPayload) -> list[str]:
    errors: list[str] = []

    lane_ids = [lane.id for lane in workflow.lanes]
    node_ids = [node.id for node in workflow.nodes]

    if not any(node.tipo == "inicio" for node in workflow.nodes):
        errors.append("Debe existir al menos un nodo tipo 'inicio'.")

    if not any(node.tipo == "fin" for node in workflow.nodes):
        errors.append("Debe existir al menos un nodo tipo 'fin'.")

    if workflow.nodes and not workflow.lanes:
        errors.append("Si hay nodos, debe haber lanes.")

    if len(set(lane_ids)) != len(lane_ids):
        errors.append("lane.id debe ser unico.")

    lane_dept_ids = [lane.departmentId for lane in workflow.lanes if lane.departmentId]
    if len(set(lane_dept_ids)) != len(lane_dept_ids):
        errors.append("No puede repetirse departmentId en mas de una lane.")

    lane_dept_by_id = {lane.id: lane.departmentId for lane in workflow.lanes}
    if len(set(node_ids)) != len(node_ids):
        errors.append("node.id debe ser unico.")

    if any(not node.id for node in workflow.nodes):
        errors.append("node.id no puede estar vacio.")

    if any(not lane.departmentId for lane in workflow.lanes):
        errors.append("Cada lane debe tener departmentId.")

    department_catalog_ids = {d.id for d in catalogs.departments}
    user_catalog_ids = {u.id for u in catalogs.users}

    for lane in workflow.lanes:
        if lane.departmentId and lane.departmentId not in department_catalog_ids:
            errors.append(f"lane '{lane.id}' usa departmentId fuera de catalogo.")

    outgoing_count: dict[str, int] = {}
    incoming_count: dict[str, int] = {}

    for edge in workflow.edges:
        if edge.fromNodeId not in node_ids:
            errors.append(f"edge.fromNodeId '{edge.fromNodeId}' no existe.")
        if edge.toNodeId not in node_ids:
            errors.append(f"edge.toNodeId '{edge.toNodeId}' no existe.")
        outgoing_count[edge.fromNodeId] = outgoing_count.get(edge.fromNodeId, 0) + 1
        incoming_count[edge.toNodeId] = incoming_count.get(edge.toNodeId, 0) + 1

    for node in workflow.nodes:
        lane_dept = lane_dept_by_id.get(node.laneId)
        if lane_dept is None:
            errors.append(f"node '{node.id}' referencia laneId inexistente '{node.laneId}'.")
        elif node.departmentId != lane_dept:
            errors.append(
                f"node '{node.id}' tiene departmentId inconsistente con su lane '{node.laneId}'."
            )

        if node.departmentId and node.departmentId not in department_catalog_ids:
            errors.append(f"node '{node.id}' usa departmentId fuera de catalogo.")

        if node.tipo == "decision":
            if not node.decisionRule:
                errors.append(f"node decision '{node.id}' debe tener decisionRule.")
            else:
                if node.decisionRule.onTrueDestinoNodeId not in node_ids:
                    errors.append(
                        f"decisionRule.onTrueDestinoNodeId invalido en node '{node.id}'."
                    )
                if node.decisionRule.onFalseDestinoNodeId not in node_ids:
                    errors.append(
                        f"decisionRule.onFalseDestinoNodeId invalido en node '{node.id}'."
                    )

        if node.tipo == "paralelo_inicio" and outgoing_count.get(node.id, 0) < 2:
            errors.append(f"node paralelo_inicio '{node.id}' debe tener >= 2 aristas salientes.")

        if node.tipo == "paralelo_fin" and incoming_count.get(node.id, 0) < 2:
            errors.append(f"node paralelo_fin '{node.id}' debe tener >= 2 aristas entrantes.")

        if node.tipo == "actividad":
            if not node.responsableTipo:
                errors.append(f"node actividad '{node.id}' debe tener responsableTipo.")
            if node.responsableTipo == "usuario" and not node.responsableUsuarioId:
                errors.append(
                    f"node actividad '{node.id}' requiere responsableUsuarioId cuando responsableTipo='usuario'."
                )
            if (
                node.responsableTipo == "usuario"
                and node.responsableUsuarioId
                and node.responsableUsuarioId not in user_catalog_ids
            ):
                errors.append(f"node actividad '{node.id}' usa responsableUsuarioId fuera de catalogo.")

        if node.form and node.form.campos:
            field_ids = [f.id for f in node.form.campos]
            if len(set(field_ids)) != len(field_ids):
                errors.append(f"node '{node.id}' tiene form.campos con id duplicado.")
            for field in node.form.campos:
                if not field.id or not field.label or not field.tipo:
                    errors.append(
                        f"node '{node.id}' tiene un campo de formulario sin id/label/tipo obligatorio."
                    )

    return errors


def _build_user_payload(body: EditWorkflowRequest) -> str:
    payload = {
        "requestId": body.requestId,
        "mode": body.mode,
        "userInstruction": body.userInstruction,
        "currentWorkflow": body.currentWorkflow.model_dump(),
        "catalogs": body.catalogs.model_dump(),
        "rules": body.rules.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False)


def _user_requested_deletions(user_instruction: str) -> bool:
    instruction = user_instruction.lower()
    deletion_keywords = [
        "eliminar",
        "borrar",
        "quitar",
        "remover",
        "reducir",
        "simplificar",
        "recortar",
        "suprimir",
        "delete",
        "remove",
        "drop",
    ]
    return any(keyword in instruction for keyword in deletion_keywords)


def _validate_id_stability(current: WorkflowModel, proposed: WorkflowModel) -> list[str]:
    errors: list[str] = []

    if current.id and proposed.id and current.id != proposed.id:
        errors.append("No se permite cambiar workflow.id existente.")

    current_lane_ids = {lane.id for lane in current.lanes}
    proposed_lane_ids = {lane.id for lane in proposed.lanes}
    missing_lane_ids = current_lane_ids - proposed_lane_ids
    if missing_lane_ids:
        errors.append(f"No se permite eliminar lanes existentes: {sorted(missing_lane_ids)}")

    current_node_ids = {node.id for node in current.nodes}
    proposed_node_ids = {node.id for node in proposed.nodes}
    missing_node_ids = current_node_ids - proposed_node_ids
    if missing_node_ids:
        errors.append(f"No se permite eliminar nodes existentes: {sorted(missing_node_ids)}")

    current_edges = {(edge.fromNodeId, edge.toNodeId, edge.tipo, edge.label) for edge in current.edges}
    proposed_edges = {(edge.fromNodeId, edge.toNodeId, edge.tipo, edge.label) for edge in proposed.edges}
    missing_edges = current_edges - proposed_edges
    if missing_edges:
        errors.append("No se permite eliminar edges existentes sin solicitud explicita.")

    return errors


def _repair_workflow_proposal(current: WorkflowModel, proposed: WorkflowModel) -> WorkflowModel:
    repaired = WorkflowModel.model_validate(proposed.model_dump())

    current_lane_by_id = {lane.id: lane for lane in current.lanes}
    repaired_lane_by_id = {lane.id: lane for lane in repaired.lanes}
    for lane_id, lane in current_lane_by_id.items():
        if lane_id not in repaired_lane_by_id:
            repaired.lanes.append(lane)

    current_node_by_id = {node.id: node for node in current.nodes}
    repaired_node_by_id = {node.id: node for node in repaired.nodes}
    for node_id, node in current_node_by_id.items():
        if node_id not in repaired_node_by_id:
            repaired.nodes.append(node)

    current_edge_by_key = {
        (edge.fromNodeId, edge.toNodeId, edge.tipo, edge.label): edge for edge in current.edges
    }
    repaired_edge_by_key = {
        (edge.fromNodeId, edge.toNodeId, edge.tipo, edge.label): edge for edge in repaired.edges
    }
    for edge_key, edge in current_edge_by_key.items():
        if edge_key not in repaired_edge_by_key:
            repaired.edges.append(edge)

    if not any(node.tipo == "inicio" for node in repaired.nodes):
        current_inicio = next((node for node in current.nodes if node.tipo == "inicio"), None)
        if current_inicio is not None:
            repaired.nodes.append(current_inicio)

    if not any(node.tipo == "fin" for node in repaired.nodes):
        current_fin = next((node for node in current.nodes if node.tipo == "fin"), None)
        if current_fin is not None:
            repaired.nodes.append(current_fin)

    if repaired.nodes and not repaired.lanes:
        repaired.lanes = list(current.lanes)

    return repaired


def _call_groq_for_workflow(body: EditWorkflowRequest) -> dict[str, Any]:
    api_key = os.getenv("API_IA")
    if not api_key:
        raise HTTPException(status_code=500, detail="Falta API_IA en variables de entorno o en .env")

    client = Groq(api_key=api_key)
    model_name = os.getenv("MODELO", "llama-3.3-70b-versatile")

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        max_completion_tokens=4096,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_payload(body),
            },
        ],
    )

    content = completion.choices[0].message.content if completion.choices else None
    if not content:
        raise HTTPException(status_code=502, detail="La IA no devolvio contenido.")

    try:
        parsed: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"La IA devolvio JSON invalido: {exc.msg}") from exc

    return parsed


def _ensure_start_and_end_nodes(current: WorkflowModel, proposed: WorkflowModel) -> WorkflowModel:
    if any(node.tipo == "inicio" for node in proposed.nodes) and any(node.tipo == "fin" for node in proposed.nodes):
        return proposed

    repaired = WorkflowModel.model_validate(proposed.model_dump())

    current_inicio = next((node for node in current.nodes if node.tipo == "inicio"), None)
    current_fin = next((node for node in current.nodes if node.tipo == "fin"), None)

    if not any(node.tipo == "inicio" for node in repaired.nodes) and current_inicio is not None:
        if all(node.id != current_inicio.id for node in repaired.nodes):
            repaired.nodes.append(current_inicio)

    if not any(node.tipo == "fin" for node in repaired.nodes) and current_fin is not None:
        if all(node.id != current_fin.id for node in repaired.nodes):
            repaired.nodes.append(current_fin)

    return repaired


def _complete_missing_required_nodes(current: WorkflowModel, proposed: WorkflowModel) -> WorkflowModel:
    repaired = WorkflowModel.model_validate(proposed.model_dump())

    if not any(node.tipo == "inicio" for node in repaired.nodes):
        current_inicio = next((node for node in current.nodes if node.tipo == "inicio"), None)
        if current_inicio is not None and all(node.id != current_inicio.id for node in repaired.nodes):
            repaired.nodes.append(current_inicio)

    if not any(node.tipo == "fin" for node in repaired.nodes):
        current_fin = next((node for node in current.nodes if node.tipo == "fin"), None)
        if current_fin is not None and all(node.id != current_fin.id for node in repaired.nodes):
            repaired.nodes.append(current_fin)

    return repaired


@fastapi_app.post("/workflow/editar")
async def edit_workflow(body: EditWorkflowRequest):
    try:
        proposal_dict = _call_groq_for_workflow(body)
        proposal_workflow = WorkflowModel.model_validate(proposal_dict)

        deletion_requested = _user_requested_deletions(body.userInstruction)

        if not deletion_requested:
            proposal_workflow = _repair_workflow_proposal(body.currentWorkflow, proposal_workflow)
            proposal_workflow = _ensure_start_and_end_nodes(body.currentWorkflow, proposal_workflow)
        else:
            proposal_workflow = _complete_missing_required_nodes(body.currentWorkflow, proposal_workflow)

        if body.rules.preserveExistingIds and not deletion_requested:
            proposal_workflow = _repair_workflow_proposal(body.currentWorkflow, proposal_workflow)

        validation_errors = _validate_workflow_structure(proposal_workflow, body.catalogs)
        if body.rules.preserveExistingIds and not deletion_requested:
            validation_errors.extend(
                _validate_id_stability(body.currentWorkflow, proposal_workflow)
            )
        if body.rules.strictValidation and validation_errors:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "La propuesta de IA no cumple reglas estructurales.",
                    "errors": validation_errors,
                },
            )

        return proposal_workflow.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en /workflow/editar: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la edicion de workflow")


class ConsultaRequest(BaseModel):
    prompt: str

@fastapi_app.post("/consultar")
async def ai_endpoint(body: ConsultaRequest):
    try:
        api_key = os.getenv("API_IA")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Falta API_IA en variables de entorno o en .env",
            )

        client = Groq(api_key=api_key)
        
        # 2. Llamada optimizada
        completion = client.chat.completions.create(
            model=os.getenv("MODELO"),
            messages=[{"role": "user", "content": body.prompt}],
            temperature=0.7,          # Un poco más bajo suele ser mejor para respuestas coherentes
            max_completion_tokens=1024,
            top_p=1,
            # Se eliminó reasoning_effort porque Llama no lo soporta
            stream=False,
        )

        respuesta = completion.choices[0].message.content
        return {"status": "success", "ia_response": respuesta}
        
    except Exception as e:
        # 3. Imprime el error en tu consola para que sepas qué pasó
        print(f"Error en el servidor: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la IA")