// Activation Functions
function relu(x) {
    return x.map(val => Math.max(0, val));
}

function sigmoid(x) {
    return x.map(val => 1 / (1 + Math.exp(-val)));
}

// Matrix multiplication + bias addition
function dotAdd(input, weights, bias) {
    let output = new Array(weights[0].length).fill(0);
    for (let j = 0; j < weights[0].length; j++) {
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * weights[i][j];
        }
        output[j] = sum + bias[j];
    }
    return output;
}

// Main Predict Function
function predictChurn(features) {
    if (!window.modelData) {
        console.error("Model data not loaded!");
        return 0;
    }
    
    let w1 = modelData.weights[0];
    let b1 = modelData.biases[0];
    let w2 = modelData.weights[1];
    let b2 = modelData.biases[1];
    let w3 = modelData.weights[2];
    let b3 = modelData.biases[2];
    
    // Layer 1
    let h1 = dotAdd(features, w1, b1);
    h1 = relu(h1);
    
    // Layer 2
    let h2 = dotAdd(h1, w2, b2);
    h2 = relu(h2);
    
    // Output Layer
    let out = dotAdd(h2, w3, b3);
    out = sigmoid(out);
    
    return out[0]; // Return the probability of leaving
}

// UI Interaction
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const resultContainer = document.getElementById("result-container");
    const resultTitle = document.getElementById("result-title");
    const resultMessage = document.getElementById("result-message");
    const probabilityFill = document.getElementById("probability-fill");
    const probabilityText = document.getElementById("probability-text");
    
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        // 1. Gather inputs in the correct order:
        // 'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 
        // 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments', 'salary'
        
        const features = [
            parseFloat(document.getElementById("satisfaction").value),
            parseFloat(document.getElementById("evaluation").value),
            parseInt(document.getElementById("projects").value, 10),
            parseInt(document.getElementById("hours").value, 10),
            parseInt(document.getElementById("time").value, 10),
            parseInt(document.getElementById("accident").value, 10),
            parseInt(document.getElementById("promotion").value, 10),
            parseInt(document.getElementById("department").value, 10),
            parseInt(document.getElementById("salary").value, 10)
        ];
        
        // 2. Predict
        const prob = predictChurn(features);
        
        // 3. Update UI
        resultContainer.classList.remove("hidden");
        
        const probPercentage = (prob * 100).toFixed(1) + "%";
        probabilityText.textContent = probPercentage;
        probabilityFill.style.width = probPercentage;
        
        if (prob > 0.5) {
            resultTitle.textContent = "High Risk of Leaving";
            resultTitle.style.color = "var(--danger)";
            resultMessage.textContent = "The model predicts that this employee is likely to churn.";
            probabilityFill.style.backgroundColor = "var(--danger)";
        } else {
            resultTitle.textContent = "Likely to Stay";
            resultTitle.style.color = "var(--success)";
            resultMessage.textContent = "The model predicts that this employee is happy and will stay.";
            probabilityFill.style.backgroundColor = "var(--success)";
        }
    });
});
