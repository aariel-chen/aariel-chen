// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Chart data
    var initialChartData = {
        labels: ['1993', '1994', '1995', '1996', '1997', '1998', '1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'],
        datasets: [
            {
                label: 'Global Land',
                data: [0.32, 0.43,0.49,0.37,0.56,0.76,0.37,0.36,0.54,0.53,0.52,0.47,0.66,0.65,0.57,0.51,0.64,0.7,0.63,0.66,0.67,0.72,0.87,0.83,0.77,0.75,0.9,0.92,0.85,0.89],
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 2,
                fill: false
            },
            {
                label: 'Global Ocean',
                data: [0.33,0.31,0.35,0.35,0.49,0.6,0.32,0.28,0.49,0.52,0.49,0.44,0.52,0.5,0.46,0.42,0.62,0.57,0.48,0.49,0.51,0.67,0.74,0.78,0.7,0.64,0.79,0.76,0.67,0.7],
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 2,
                fill: false
            }
        ]
    };

    var alternateChartData = { 
        labels: ['1993', '1994', '1995', '1996', '1997', '1998', '1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'],
        datasets: [
            {
                label: 'Asia Land',
                data: [0.25,0.78,0.74,0.18,0.52,1.06,0.33,1.02,1.02,0.92,0.47,0.61,1.27,1.37,0.8,1.01,0.8,1.48,1.46,1.87,1.05,1.08,1.42,1.42,1.23,1.44,1.43,1.29,1.66,1.83],
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 2,
                fill: false
            },
            {
                label: 'North America Land',
                data: [-0.01,0.93,0.77,0.7,0.59,0.95,0.42,-0.01,0.48,0.94,0.33,0.21,0.83,1.31,0.79,0.8,0.13,1.08,0.85,1.54,1.33,0.85,1.37,1.53,0.98,1.01,0.79,0.92,1.78,1.18],
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 2,
                fill: false
            }
        ] 
    };
    var currentChartType = "initial";


    // Chart configuration
    var initialChartConfig = {
        type: 'line',
        data: initialChartData,
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Year',
                        color: 'black',
                        font:{
                            size:16
                        }
                    },
                    grid:{
                        color: "black"
                    },
                    ticks:{
                        color: 'black',
                        font:{
                            size:14
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temperature',
                        color: 'black',
                        font:{
                            size: 16
                        }
                    },
                    grid:{
                        color: "black"
                    },
                    ticks:{
                        color: 'black',
                        font:{
                            size:14
                        }
                    }
                }                
            },
            plugins:{
                tooltip:{
                    enabled: true,
                    intersect: false
                }
            },
            backgroundColor: "white"
        }
    };
    
    var alternateChartConfig = {
        type: 'line',
        data: alternateChartData,
        options: {
            responsive: true,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Year',
                        color: 'black',
                        font:{
                            size:16
                        }
                    },
                    grid:{
                        color: "black"
                    },
                    ticks:{
                        color: 'black',
                        font:{
                            size:14
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temperature',
                        color: 'black',
                        font:{
                            size: 16
                        }
                    },
                    grid:{
                        color: "black"
                    },
                    ticks:{
                        color: 'black',
                        font:{
                            size:14
                        }
                    }
                }                
            },
            plugins:{
                tooltip:{
                    enabled: true,
                    intersect: false
                }
            },
            backgroundColor: "white"
        }
    };

    

    // Create the chart
    var lineChart = new Chart(document.getElementById('lineChart'), initialChartConfig);
    
    document.getElementById('toggleButton').addEventListener('click', function() {
        if (currentChartType === 'initial') {
            lineChart.destroy(); // Destroy the existing chart
            lineChart = new Chart(document.getElementById('lineChart'), alternateChartConfig); // Create the alternate chart
            currentChartType = 'alternate'; // Set the current chart type to 'alternate'
        } else {
            lineChart.destroy(); // Destroy the existing chart
            lineChart = new Chart(document.getElementById('lineChart'), initialChartConfig); // Create the initial chart
            currentChartType = 'initial'; // Set the current chart type to 'initial'
        }
    });
});
