document.addEventListener('DOMContentLoaded', () => {
    const structureSelect = document.getElementById('structure-select');
    const generateBtn = document.getElementById('generate-btn');
    const instanceInfo = document.getElementById('instance-info');
    const resultsContainer = document.getElementById('results-container');
    let cityChart = null;

    // Fetch structures on load
    fetch('/api/structures')
        .then(response => response.json())
        .then(structures => {
            structures.forEach(structure => {
                const option = document.createElement('option');
                option.value = structure;
                option.textContent = structure.charAt(0).toUpperCase() + structure.slice(1).replace('_', ' ');
                structureSelect.appendChild(option);
            });

            // Auto-select 'grid' and load a random instance
            if (structures.includes('grid')) {
                structureSelect.value = 'grid';
                // Automatically generate an instance
                generateBtn.click();
            }
        });

    generateBtn.addEventListener('click', () => {
        const structure = structureSelect.value;
        if (!structure) {
            alert('Please select a structure type first.');
            return;
        }

        // Fetch random instance
        fetch(`/api/random_instance/${structure}`)
            .then(response => {
                if (!response.ok) throw new Error('Failed to fetch instance');
                return response.json();
            })
            .then(instance => {
                updateInstanceInfo(instance);
                fetchInstanceData(instance);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error generating instance. Please try again.');
            });
    });

    function updateInstanceInfo(instance) {
        instanceInfo.innerHTML = `
            <strong>Instance:</strong> ${instance.instance} (n=${instance.n_cities})<br>
            <strong>CV:</strong> ${parseFloat(instance.cv).toFixed(3)}
        `;
    }

    function fetchInstanceData(instance) {
        // We need to pass the file names to get the data
        const params = new URLSearchParams({
            coord_file: instance.coord_file,
            dist_file: instance.dist_file,
            n: instance.n_cities,
            structure: instance.structure,
            instance_idx: instance.instance
        });

        fetch(`/api/instance_data?${params}`)
            .then(response => response.json())
            .then(data => {
                renderChart(data.coordinates, data.tour);
                renderResults(data.results);
            });
    }

    function renderChart(coordinates, tour) {
        const ctx = document.getElementById('city-chart').getContext('2d');
        const dataPoints = coordinates.map(c => ({ x: c.x, y: c.y }));

        // Prepare tour dataset if tour exists
        let tourData = [];
        if (tour && tour.length > 0) {
            // Create ordered list of points based on tour indices
            tourData = tour.map(index => dataPoints[index]);
            // Close the loop
            tourData.push(dataPoints[tour[0]]);
        }

        if (cityChart) {
            cityChart.destroy();
        }

        cityChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Optimal Tour',
                        data: tourData,
                        showLine: true,
                        borderColor: '#22c55e', // Green color for the tour
                        borderWidth: 2,
                        pointRadius: 0, // Hide points for the line dataset
                        fill: false,
                        order: 2
                    },
                    {
                        label: 'Cities',
                        data: dataPoints,
                        backgroundColor: '#38bdf8',
                        borderColor: '#38bdf8',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#f8fafc' }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.dataset.label === 'Cities') {
                                    return `(${context.parsed.x.toFixed(1)}, ${context.parsed.y.toFixed(1)})`;
                                }
                                return null;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderResults(results) {
        resultsContainer.innerHTML = '';

        if (results.length === 0) {
            resultsContainer.innerHTML = '<p class="placeholder-text">No results found for this instance.</p>';
            return;
        }

        results.forEach(res => {
            // Distinguish between zero gap and missing data
            // If gap_absolute is 0, it's truly zero (optimal solution)
            // If gap_percent is null but gap_absolute is not 0, data is missing
            let gapPercent;
            if (res.gap_absolute !== null && res.gap_absolute !== undefined && Math.abs(res.gap_absolute) < 0.0001) {
                gapPercent = '0.0000';
            } else if (res.gap_percent !== null && res.gap_percent !== undefined) {
                gapPercent = parseFloat(res.gap_percent).toFixed(4);
            } else {
                gapPercent = 'N/A';
            }
            const div = document.createElement('div');
            div.className = 'result-item';
            div.innerHTML = `
                <div class="result-header">
                    <span class="algo-name">${res.algorithm}</span>
                    <span class="gap-badge">Gap: ${gapPercent}%</span>
                </div>
                <div class="result-details">
                    <div class="detail-row">
                        <span>IP Obj:</span>
                        <span>${parseFloat(res.ip_obj).toFixed(2)}</span>
                    </div>
                    <div class="detail-row">
                        <span>LP Obj:</span>
                        <span>${parseFloat(res.lp_obj).toFixed(2)}</span>
                    </div>
                    <div class="detail-row">
                        <span>Solve Time:</span>
                        <span>${parseFloat(res.solve_time).toFixed(3)}s</span>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(div);
        });
    }
});
