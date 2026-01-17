import React, { useState } from "react";

const ConfigForm: React.FC = () => {
  const [example, setExample] = useState("");

  return (
    <form>
      <label>
        Example setting:
        <input
          type="text"
          value={example}
          onChange={(e) => setExample(e.target.value)}
        />
      </label>
    </form>
  );
};

export default ConfigForm;