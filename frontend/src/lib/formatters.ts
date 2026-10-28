export const percent = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
export const number = (value: number | null | undefined, digits = 2) => value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
export const label = (value: string) => value.replaceAll("___", " · ").replaceAll("_", " ");
export const modelLabel = (name: string) => name === "resnet18" ? "ResNet-18" : name === "mobilenet_v3_small" ? "MobileNetV3-Small" : name;
export const thresholdLabel = (value: number | null) => value !== null && value > 1 ? "Abstain on all images" : percent(value);
