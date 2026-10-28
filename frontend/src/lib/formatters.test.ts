import { it, expect } from "vitest";
import { percent, number } from "./formatters";
it("distinguishes undefined metrics from zero", () => { expect(percent(null)).toBe("—"); expect(percent(0)).toBe("0.0%"); expect(number(undefined)).toBe("—"); });
