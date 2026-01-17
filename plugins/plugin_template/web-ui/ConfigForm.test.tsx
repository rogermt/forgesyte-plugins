import { render, screen } from "@testing-library/react";
import ConfigForm from "./ConfigForm";

test("renders config form input", () => {
  render(<ConfigForm />);
  expect(screen.getByLabelText(/Example setting/i)).toBeInTheDocument();
});