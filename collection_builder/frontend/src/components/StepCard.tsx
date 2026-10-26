import type { ParamSchema } from "../types";
import type { StepInstance } from "../compose";

/** A field is a string box unless its schema says otherwise; `anyOf` means nullable. */
function fieldType(schema: ParamSchema): "number" | "checkbox" | "text" {
  const t = schema.type ?? schema.anyOf?.find((a) => a.type !== "null")?.type;
  if (t === "integer" || t === "number") return "number";
  if (t === "boolean") return "checkbox";
  return "text";
}

type FieldProps = {
  name: string;
  schema: ParamSchema;
  value: unknown;
  onChange: (v: unknown) => void;
};

function Field({ name, schema, value, onChange }: FieldProps) {
  const kind = fieldType(schema);
  const placeholder =
    schema.default === null || schema.default === "" ? "—" : String(schema.default);

  if (kind === "checkbox") {
    return (
      <label className="step-field step-field-inline">
        <input
          type="checkbox"
          checked={Boolean(value ?? schema.default)}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>{name}</span>
      </label>
    );
  }
  return (
    <label className="step-field">
      <span>{name}</span>
      <input
        type={kind === "number" ? "number" : "text"}
        value={value === undefined || value === null ? "" : String(value)}
        placeholder={placeholder}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return onChange(undefined);
          onChange(kind === "number" ? Number(raw) : raw);
        }}
      />
    </label>
  );
}

type Props = {
  step: StepInstance;
  schema: Record<string, ParamSchema>;
  kind: "study" | "transform";
  subtitle?: string;
  badges?: string[];
  onChangeParam: (key: string, value: unknown) => void;
  onRemove?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
};

export function StepCard({
  step,
  schema,
  kind,
  subtitle,
  badges,
  onChangeParam,
  onRemove,
  onMoveUp,
  onMoveDown,
}: Props) {
  const params = Object.entries(schema);
  return (
    <div className={`step-card step-${kind}`}>
      <div className="step-head">
        <div className="step-title">
          <span className="step-kind">{kind}</span>
          <code>{step.name}</code>
        </div>
        {onRemove && (
          <div className="step-actions">
            <button type="button" onClick={onMoveUp} disabled={!onMoveUp} title="move up">
              ▲
            </button>
            <button
              type="button"
              onClick={onMoveDown}
              disabled={!onMoveDown}
              title="move down"
            >
              ▼
            </button>
            <button type="button" onClick={onRemove} title="remove">
              ✕
            </button>
          </div>
        )}
      </div>

      {subtitle && <p className="step-sub">{subtitle}</p>}
      {badges && badges.length > 0 && (
        <div className="step-badges">
          {badges.map((b) => (
            <span key={b} className="step-badge">
              {b}
            </span>
          ))}
        </div>
      )}

      {params.length > 0 && (
        <div className="step-params">
          {params.map(([key, sch]) => (
            <Field
              key={key}
              name={key}
              schema={sch}
              value={step.params[key]}
              onChange={(v) => onChangeParam(key, v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
